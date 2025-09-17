# This is based on Mem0 (https://github.com/mem0ai/mem0/tree/main/evaluation/src/memzero)

import asyncio
import json
import os
import subprocess
import time
from collections import defaultdict

from dotenv import load_dotenv
from httpx import AsyncClient
from locomo_agent import locomo_response
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

load_dotenv()
root_dir = os.path.dirname(os.path.abspath(__file__))

# Global Variables
# commit_id will remain the same for runs as long as git commit id remains the same
commit_id = (
    subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"])
    .decode("utf-8")
    .strip()
)


class MemMachineAdd:
    def __init__(self, data_path=None, batch_size=1, reprocess=False):
        self.batch_size = batch_size
        self.semaphores = defaultdict(
            lambda: asyncio.Semaphore(self.batch_size)
        )
        self.data_path = data_path
        self.data = None
        self.reprocess = reprocess
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    async def add_memory(
        self,
        idx,
        message_data,
        base_url: str = "http://localhost:8080",
        retries=3,
    ):
        """
        Add memory using HTTP POST to MemMachine API.

        Args:
            message_data: Dictionary containing user_id, query, and metadata
            base_url: The base URL of the MemMachine API (default: http://localhost:8080)
            retries: Number of retry attempts on failure

        Returns:
            bool: True if successful, False otherwise
        """
        async with self.semaphores[idx]:
            for attempt in range(retries):
                try:
                    async with AsyncClient(
                        base_url=base_url, timeout=60
                    ) as client:
                        headers = {"Content-Type": "application/json"}
                        response = await client.post(
                            "/v1/memories", json=message_data, headers=headers
                        )

                except Exception as e:
                    print(f"Error adding memory: {e}")
                    print(f"Error adding memory: {message_data}")
                    if attempt < retries - 1:
                        time.sleep(1)  # Wait before retrying
                        continue
                    return False

                if response.status_code == 200:
                    return True
                else:
                    print(
                        f"Failed to add memory. Status code: {response.status_code}, Response: {response.text}"
                    )
                    if attempt < retries - 1:
                        time.sleep(1)  # Wait before retrying
                        continue
                    return False

            return False

    async def add_memories_for_speaker(self, idx, messages, desc, base_url):
        for message in messages:
            await atqdm.gather(
                *[self.add_memory(idx, message, base_url=base_url)], desc=desc
            )

    async def process_conversation(self, item, idx, base_url):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        print(f"Speaker A: {speaker_a}, Conversation: {idx}")
        print(f"Speaker B: {speaker_b}, Conversation: {idx}")

        session_date_time = None
        session_count = -1
        total_session_count = sum(
            1 for k in conversation.keys() if isinstance(conversation[k], list)
        )

        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "timestamp" in key:
                continue

            # Get session date time
            if "date_time" in key:
                session_date_time = conversation[key]
                session_count = key.split("_")[1]
                continue

            chats = conversation[key]

            messages = []

            # Length of messages will be number of session in conversation
            for chat in chats:
                if chat["speaker"] == speaker_a:
                    messages.append(
                        {
                            "producer": f"{speaker_a}",
                            "produced_for": f"{speaker_a}",
                            "episode_content": chat["text"],
                            "episode_type": "default",
                            "metadata": {
                                "speaker": speaker_a,
                                "timestamp": session_date_time,
                                "blip_caption": chat.get("blip_caption"),
                            },
                            "session": {
                                "group_id": f"{idx}",
                                "user_id": [speaker_a, speaker_b],
                                "agent_id": [],
                                "session_id": f"{idx}",
                            },
                        }
                    )
                elif chat["speaker"] == speaker_b:
                    messages.append(
                        {
                            "producer": f"{speaker_b}",
                            "produced_for": f"{speaker_b}",
                            "episode_content": chat["text"],
                            "episode_type": "default",
                            "metadata": {
                                "speaker": speaker_b,
                                "timestamp": session_date_time,
                                "blip_caption": chat.get("blip_caption"),
                            },
                            "session": {
                                "group_id": f"{idx}",
                                "user_id": [speaker_a, speaker_b],
                                "agent_id": [],
                                "session_id": f"{idx}",
                            },
                        }
                    )
                else:
                    raise ValueError(f"Unknown speaker: {chat['speaker']}")

            await self.add_memories_for_speaker(
                idx,
                messages,
                f"Doing session {session_count} of {total_session_count} for {speaker_a} and {speaker_b}",
                base_url,
            )

            print("Messages added successfully")

    async def process_all_conversations(
        self, max_samples=None, base_url="http://localhost:8080"
    ):
        if not self.data:
            raise ValueError(
                "No data loaded. Please set data_path and call load_data() first."
            )

        await atqdm.gather(
            *[
                self.process_conversation(item, idx, base_url)
                for idx, item in enumerate(self.data)
            ]
        )


class MemMachineSearch:
    def __init__(self, output_path=f"results_IM_{commit_id}.json"):
        self._semaphore = asyncio.Semaphore(10)
        self._output_path = output_path
        if not os.path.exists(output_path):
            self.results = defaultdict(list)
        else:
            with open(output_path, "r") as f:
                self.results = defaultdict(list, json.load(f))

    async def answer_question_with_mcp(self, question, idx, users):
        t1 = time.time()
        response_parsed = await locomo_response(
            idx, question, users, "gpt-4o-mini"
        )
        t2 = time.time()
        return (
            response_parsed["response"],
            response_parsed["trace"],
            t2 - t1,
        )

    async def process_question(self, val, idx, users, base_url):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            trace,
            response_time,
        ) = await self.answer_question_with_mcp(question, idx, users)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "agent_trace": trace,
            "response_time": response_time,
        }

        return result

    async def process_data_file(
        self, file_path, exclude_category={5}, base_url="http://localhost:8080"
    ):
        async def process_item(idx, item):
            if str(idx) in self.results.keys():
                print(f"Conversation {idx} has already been processed")
                return
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]
            qa_filtered = [
                i for i in qa if i.get("category", -1) not in exclude_category
            ]

            print(
                f"Filter category: {exclude_category}, {len(qa)} -> {len(qa_filtered)}"
            )

            results_single_convo = await self.process_questions_parallel(
                qa_filtered,
                idx,
                users=[speaker_a, speaker_b],
                base_url=base_url,
            )
            self.results[idx] = results_single_convo
            with open(self._output_path, "w") as f:
                json.dump(self.results, f, indent=4)

        with open(file_path, "r") as f:
            data = json.load(f)
            for idx, convo in tqdm(zip(range(len(data)), data)):
                await process_item(idx, convo)
        # Final save at the end
        with open(self._output_path, "w") as f:
            json.dump(self.results, f, indent=4)

    async def process_questions_parallel(
        self, qa_list, idx, users, base_url="http://localhost:8080"
    ):
        async def process_single_question(val):
            async with self._semaphore:
                result = await self.process_question(val, idx, users, base_url)
                return result

        out = await atqdm.gather(
            *[process_single_question(val) for val in qa_list],
            desc="processing questions",
        )
        return list(out)
