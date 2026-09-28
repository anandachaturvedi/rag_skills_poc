import os
import re
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# ============================================================
# SETUP
# ============================================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

MODEL = "openai/gpt-4o-mini"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2") #produces a 384-dimensional vector

SKILLS_DIR = Path("skills")
DOCUMENTS_DIR = Path("documents")


# ============================================================
# 1. DISCOVER SKILL METADATA
# ============================================================

def load_skill_metadata():

    skills = []

    for skill_file in SKILLS_DIR.glob("*/skills.md"):

        content = skill_file.read_text(
            encoding="utf-8"
        )

        # Read YAML front matter
        match = re.match(
            r"^---\s*\n(.*?)\n---",
            content,
            re.DOTALL
        )

        if not match:
            continue

        frontmatter = match.group(1)

        # Get name
        name_match = re.search(
            r"^name:\s*(.+)$",
            frontmatter,
            re.MULTILINE
        )

        # Get description
        description_match = re.search(
            r"description:\s*>-\s*\n((?:\s+.*\n?)+)",
            frontmatter
        )

        # Get document source
        source_match = re.search(
            r"^knowledge_source:\s*(.+)$",
            frontmatter,
            re.MULTILINE
        )

        if not name_match:
            continue

        name = name_match.group(1).strip()

        if description_match:
            description = " ".join(
                line.strip()
                for line in description_match.group(1).splitlines()
                if line.strip()
            )
        else:
            description = ""

        if source_match:
            knowledge_source = source_match.group(1).strip()
        else:
            knowledge_source = ""

        skills.append({
            "name": name,
            "description": description,
            "knowledge_source": knowledge_source,
            "skill_file": str(skill_file)
        })

    return skills


# ============================================================
# 2. LOAD SKILLS
# ============================================================

skills = load_skill_metadata()


# ============================================================
# 3. DISPLAY WHAT WAS DISCOVERED
# ============================================================

print("\n=== Available Skills ===")

for skill in skills:

    print("\nSkill:")
    print(skill["name"])

    print("Description:")
    print(skill["description"])

    print("Knowledge source:")
    print(skill["knowledge_source"])

# ============================================================
# 4. PREPARE SKILL METADATA FOR THE LLM
# ============================================================

skill_metadata = "\n\n".join(
    f"Skill: {skill['name']}\n"
    f"Description: {skill['description']}"
    for skill in skills
)


# ============================================================
# 5. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are a skill-routing assistant.

Your job is to determine which skill is most relevant
to the user's question.

Available skills:

{skill_metadata}

Choose the ONE most relevant skill.

Do not answer the user's question yet.
Instead, select the appropriate skill.
"""


# ============================================================
# 6. USER QUERY
# ============================================================

QUERY = "What did the pocket watch do?"


# ============================================================
# 7. ASK THE LLM TO SELECT A SKILL
# ============================================================

response = client.chat.completions.create(

    model=MODEL,

    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": QUERY
        }
    ]
)


# ============================================================
# 8. PRINT LLM'S DECISION
# ============================================================

selected_skill = response.choices[0].message.content

print("\n=== LLM Selected Skill ===")
print(selected_skill)


# ============================================================
# 9. TOKEN USAGE
# ============================================================

print("\n=== Router Token Usage ===")

print(
    "Prompt tokens:",
    response.usage.prompt_tokens
)

print(
    "Completion tokens:",
    response.usage.completion_tokens
)

print(
    "Total tokens:",
    response.usage.total_tokens
)

# ============================================================
# 10. FIND THE SELECTED SKILL
# ============================================================

# Remove any extra text from the LLM response
selected_skill_name = (
    selected_skill
    .replace("Skill:", "")
    .strip()
)

selected_skill_data = next(
    (
        skill
        for skill in skills
        if skill["name"] == selected_skill_name
    ),
    None
)


if selected_skill_data is None:

    print("\nSkill not found!")

else:

    knowledge_source = selected_skill_data["knowledge_source"]

    print("\n=== Selected Knowledge Source ===")
    print(knowledge_source)

# ============================================================
# 11. READ THE KNOWLEDGE SOURCE
# ============================================================

if selected_skill_data is not None:

    knowledge_path = Path(selected_skill_data["knowledge_source"])

    document_text = knowledge_path.read_text(
        encoding="utf-8"
    )

    print("\n=== Knowledge Source Loaded ===")
    print("Characters:", len(document_text))

# ============================================================
# 12. SPLIT DOCUMENT INTO CHUNKS
# ============================================================

if selected_skill_data is not None:

    chunk_size = 800
    overlap = 200

    chunks = []

    start = 0

    while start < len(document_text):

        end = start + chunk_size

        chunks.append(document_text[start:end])

        start += chunk_size - overlap

    print("\n=== Document Chunks ===")
    print("Number of chunks:", len(chunks))

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {len(chunk)} characters")

chunk_embeddings = embedding_model.encode(
    chunks,
    convert_to_tensor=True
)

print("\n=== Chunk Embeddings Created ===")
print("Number of embeddings:", len(chunk_embeddings))
print("Embedding size:", chunk_embeddings.shape[1])

query_embedding = embedding_model.encode(      #embedding for the user query
    QUERY,
    convert_to_tensor=True
)

print("\n=== Query Embedding Created ===")
print("Embedding size:", query_embedding.shape[0])

similarities = cos_sim(         #semantic search
    query_embedding,
    chunk_embeddings
)

print("\n=== Tensor Shapes ===")
print("Query embedding shape:", query_embedding.shape)
print("Chunk embeddings shape:", chunk_embeddings.shape)
print("Similarity shape:", similarities.shape)

top_k = 2

top_scores, top_indices = similarities.topk(top_k)

top_scores = top_scores[0]
top_indices = top_indices[0]

retrieved_chunks = []

for score, index in zip(top_scores, top_indices):

    retrieved_chunks.append(chunks[index.item()])

    print("\n=== Retrieved Chunk ===")
    print("Chunk number:", index.item() + 1)
    print("Similarity score:", score.item())
    print(chunks[index.item()])

# ============================================================
# 14. GENERATE ANSWER FROM RETRIEVED CHUNK
# ============================================================

if retrieved_chunks:

    final_prompt = f"""
Answer the user's question using only the retrieved context.

Retrieved context:
{"".join(retrieved_chunks)}

User question:
{QUERY}

Give a short and direct answer.
"""

    final_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": final_prompt
            }
        ]
    )

    answer = final_response.choices[0].message.content

    print("\n=== Final Answer ===")
    print(answer)

    print("\n=== Final Answer Token Usage ===")
    print("Prompt tokens:", final_response.usage.prompt_tokens)
    print("Completion tokens:", final_response.usage.completion_tokens)
    print("Total tokens:", final_response.usage.total_tokens)