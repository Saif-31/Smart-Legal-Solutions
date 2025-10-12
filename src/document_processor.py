import logging
from src.agents import (
    legal_summary_agent,
    legal_appeal_agent,
    legal_review_agent,
    legal_lawsuit_agent,
    legal_lawsuit_response_agent,
    legal_contract_analysis_agent,
    legal_chat_helper_agent
)

class LegalDocumentProcessor:
    async def process_document(self, document: str, request_type: str, question: str = None) -> dict:
        try:
            if request_type == "summary":
                result = await legal_summary_agent(document)
            elif request_type == "appeal":
                result = await legal_appeal_agent(document)
            elif request_type == "review":
                result = await legal_review_agent(document)
            elif request_type == "lawsuit":
                result = await legal_lawsuit_agent(document)
            elif request_type == "lawsuit_response":
                result = await legal_lawsuit_response_agent(document)
            elif request_type == "contract_analysis":
                result = await legal_contract_analysis_agent(document)
            elif request_type == "chat":
                result = await legal_chat_helper_agent(document, question)
            else:
                return {"error": "Invalid request type"}

            return {"result": result}
        except Exception as e:
            return {"error": str(e)}
