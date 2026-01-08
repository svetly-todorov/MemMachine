"""Prompt templates used by the semantic memory pipeline."""


def build_update_prompt(*, tags: dict[str, str], description: str = "") -> str:
    """Create an update prompt for extracting profile changes from a query."""
    tags_list = "\n".join([f"- {key}: {value}" for key, value in tags.items()])
    
    return (
        f"""Extract user information from the query and update the profile.

Profile structure: tag -> feature -> value (two-level key-value store)

Available tags:
{tags_list}

"""
        + (description + "\n\n" if description else "")
        + """Rules:
- Each entry = one fact. Keep entries short.
- Feature names: brief summary. Value: full details.
- Extract multiple facts separately, not combined.
- Make inferences when information is implied.
- Only use tags listed above. You can create new features.
- Return {{}} only if query has zero personal information.

Output format (JSON with commands list):
{{
    "commands": [
        {{"command": "add", "tag": "Demographic Information", "feature": "name", "value": "Alice"}},
        {{"command": "delete", "tag": "Language Preferences", "feature": "format"}}
    ]
}}

Example:
Query: "Hi, I'm Sarah and I love Python programming"
{{
    "commands": [
        {{"command": "add", "tag": "Demographic Information", "feature": "name", "value": "Sarah"}},
        {{"command": "add", "tag": "Hobbies & Interests", "feature": "programming_language", "value": "Python"}}
    ]
}}

Output valid JSON only."""
    )


def build_consolidation_prompt() -> str:
    """Create a consolidation prompt for merging overlapping memories."""
    return """Merge similar memories to reduce duplication and organize information.

Input: memories with fields: tag, feature, value, metadata.id

Rules:
- Split memories that combine unrelated ideas into separate memories
- Delete redundant memories (same info already processed)
- Merge similar memories: if same tag/feature but different values, combine into one
- For lists: only merge 3+ similar memories. Don't force grouping.
- Keep feature names brief. Put details in value.
- Don't create new tag names. Use existing tags only.
- With many memories, be more aggressive about deletion

Output format:
{{
    "consolidate_memories": [
        {{"tag": "Demographic Information", "feature": "name", "value": "Alice", "metadata": {{"citations": [1, 2]}}}}
    ],
    "keep_memories": [3, 4]
}}

- consolidate_memories: new merged memories (include citations from source IDs)
- keep_memories: IDs of original memories to keep unchanged
- If no changes needed: return empty arrays

Output valid JSON only."""
