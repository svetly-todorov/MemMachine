# Benchmark Evaluations: A Guide to Testing Your MemMachine

Welcome to the MemMachine evaluation toolsets! We've created two simple tools to help you measure the performance and response quality of your MemMachine instance. You can use both of these to get a LoCoMo score for your system.

- **Episodic Memory Tool Set:** This tool measures how fast and accurately MemMachine performs core episodic memory tasks. For a list of specific commands, check out the [Episodic Memory Tool Set](./locomo/episodic_memory/README.md).
- **Episodic Profile Agent Tool Set:** This tool is designed to evaluate the speed and quality of MemMachine's Profile Agent. For a list of specific commands, check out the [Episodic Profile Agent Tool Set](./locomo/episodic_profile_agent/README.md).

## Getting Started

Before you run any benchmarks, you'll need to set up your environment.

**General Prerequisites:**

- **MemMachine Backend:** Both tools require that your MemMachine backend be installed and configured. If you need help with this, you can check out our [QuickStart Guide](http://docs.memmachine.ai/getting_started/quickstart).

- **Start the Backend:** Once everything is set up, start MemMachine with this command:

  ```sh
  memmachine-server
  ```

**Tool-Specific Prerequisites:**

- **Episodic Memory:** For this tool, please ensure your `cfg.yml` file has been copied into your `locomo` directory (`/memmachine/evaluation/locomo/`) and renamed to `locomo_config.yaml`.
- **Episodic Profile Agent:** This tool requires your MCP Server to be running before you run any commands.

## Running the Benchmark

Ready to go? Follow these simple steps. 

A. All commands should be run from their respective tool directory (e.g., `locomo/episodic_memory/` or `locomo/episodic_profile_agent/`).

B. The path to your data file, `locomo10.json`, should be updated to match its location. By default, you can find it in `/memmachine/evaluation/locomo/`.

C. Once you have performed step 1, you can repeat the benchmark run by performing steps 2-4.  Once are you finished performing the benchmark, run step 5.

### Step 1: Ingest a Conversation

First, let's add conversation data to MemMachine. This only needs to be done once per test run.

```sh
python locomo_ingest.py --data-path path/to/locomo10.json
```

### Step 2: Search the Conversation

Now, let's search through the data you just added.

```sh
python locomo_search.py --data-path path/to/locomo10.json --target-path results.json
```

### Step 3: Evaluate the Responses

Next, run a LoCoMo evaluation against the search results.

```sh
python locomo_evaluate.py --data-path results.json --target-path evaluation_metrics.json
```

### Step 4: Generate Your Final Score

Once the evaluation is complete, you can generate the final scores.

```sh
python generate_scores.py
```

The output will be a table in your shell showing the mean scores for each category and an overall score.

### Step 5: Clean Up Your Data

When you're finished, you may want to delete the test data. This is especially important before running a different benchmark.

- **For Episodic Memory:** Simply run this command:

  ```sh
  python locomo_delete.py --data-path path/to/locomo10.json
  ```

- **For Episodic Profile Agent:** You'll need to run two commands to ensure all data is removed:

  ```sh
  memmachine-sync-profile-schema --delete
  ```
Be sure to include each of the following flags for removal:
```sh
-- host <HOST> \ # or use environment variable POSTGRES_HOST
    -- port <PORT> \ # or use environment variable POSTGRES_PORT
    -- user <USER> \ # or use environment variable POSTGRES_USER
    -- password <PASSWORD> \ # or use environment variable POSTGRES_PASSWORD
    -- database <DATABASE> \ # or use environment variable POSTGRES_DB
```

  Then, clean up the `locomo` data as well:

  ```sh
  python locomo_delete.py --data-path path/to/locomo10.json
  ```
