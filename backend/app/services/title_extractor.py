import httpx
import logging
from backend.app.config import settings

logger = logging.getLogger("app.title_extractor")

async def extract_title_with_qwen2_5_0_5b(first_pages_text: str) -> str:
    """
    Extracts a concise document title from the first few pages of text using Qwen2.5 0.5B.
    """
    if not first_pages_text or not first_pages_text.strip():
        return ""

    url = f"{settings.llm.ollama_url}/api/chat"
    system_prompt = (
        "You are an expert document analyzer. "
        "Extract a short, descriptive title from the provided text of a regulatory or compliance document. "
        "Return ONLY a single title string, without quotes, numbering, or any additional explanation. "
        "Prefer concise heading-like titles that capture the main subject. "
        "If the text is mostly metadata or file-name-like, infer a more meaningful document title from the overall topic. "
        "If you cannot determine a good title, return an empty string."
    )
    
    payload = {
        "model": "qwen2.5:0.5b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"First pages text:\n{first_pages_text[:3000]}"}
        ],
        "options": {
            "temperature": 0.0
        },
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                result = response.json()["message"]["content"].strip()
                # Remove surrounding quotes if model outputs them
                result = result.strip("\"'")
                logger.info(f"[Title Extractor] Qwen2.5 0.5B generated title: {result}")
                return result
            else:
                logger.warning(f"Ollama title extraction failed with status {response.status_code}")
                return ""
    except Exception as e:
        logger.error(f"Error during LLM title extraction: {e}")
        return ""
