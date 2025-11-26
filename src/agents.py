import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize model with environment variable
model = ChatOpenAI(
    model="gpt-4o-mini",  
    openai_api_key=api_key,
    temperature=0.7,
    max_completion_tokens=2048
)

# Define prompt templates and message creation
def create_messages(prompt: str, document: str):
    return [
        SystemMessage(content="You are a legal expert AI assistant."),
        HumanMessage(content=prompt.format(document=document))
    ]

def chunk_document(document: str, max_length: int = 5000) -> list[str]:
    chunks = []
    start = 0
    while start < len(document):
        end = min(start + max_length, len(document))
        chunks.append(document[start:end])
        start = end
    return chunks

async def legal_summary_agent(document: str) -> str:
    """Generate a document summary following Serbian legal standards."""
    try:
        doc_chunks = chunk_document(document)
        summaries = []
        for chunk in doc_chunks:
            messages = create_messages(
                """You are an expert legal AI assistant specialized in Serbian law. Your primary task is to create SHORT, HIGH-EFFICIENCY summaries of legal documents. Every summary must be concise and focused only on the most critical information a lawyer needs to know.

                CORE REQUIREMENTS:

                Maximum length: 600 words total
                Focus on actionable information
                Prioritize only the most critical points
                Use precise, economical language

                SUMMARY STRUCTURE:

                BASIC INFORMATION (2–3 lines)
                Case/Document Number: [Number, Date, Type]
                Parties: [Only main parties]
                Forum: [Court/Authority]

                CRITICAL OVERVIEW (30–40 words)
                One short paragraph covering the key issue and current status.

                KEY LEGAL ELEMENTS
                Primary Legal Issue: [Most important legal question]
                Essential Facts:
                • [Max 3 bullet points]
                Decisive Arguments:
                • [Strongest argument for each side]
                Key Evidence:
                • [Only evidence that determines the case outcome]

                OUTCOME & IMPACT (2–3 bullets)
                • Decision/Status
                • Urgent required action
                • Main risk/opportunity

                VITAL REFERENCES
                • Primary legal provision
                • Precedent (if applicable)

                WRITING GUIDELINES:
                Use short, declarative sentences
                Include only information that affects decision-making
                Exclude background unless essential
                Focus on conclusions rather than explanations
                Highlight only time-sensitive elements

                Please deliver a short summary of the following document, strictly following the length and format requirements above:
                {document}
                
                Always respond in English, regardless of the document language.""",
                chunk
            )

            response = model.invoke(messages)
            summaries.append(response.content)
        return " ".join(summaries)
    except Exception as e:
        logging.error(f"Error in summary agent: {e}")
        return f"Error generating summary: {str(e)}"

async def legal_appeal_agent(document: str) -> str:
    """Generate a formal appeal based on Serbian legal standards."""
    try:
        doc_chunks = chunk_document(document)
        appeal_parts = []
        for chunk in doc_chunks:
            messages = create_messages(
                """You are a legal assistant specialized in drafting formal appeals based on the provided legal document.
                Analyze the document and generate an appeal following the structure below:

                1. Header
                [NAME OF COURT]
                [JURISDICTION]
                [Case Number]
                [NAME OF APPELLANT], Appellant
                vs.
                [NAME OF RESPONDENT], Respondent

                2. APPEAL / NOTICE OF APPEAL
                [Formal notice of appeal]

                3. Statement of Jurisdiction
                [Explanation of the court’s jurisdiction]

                4. Statement of Facts
                [Factual background]

                5. Issues on Appeal
                [List of specific issues being appealed]

                6. Argument
                [Detailed arguments for each issue]

                7. Conclusion
                [Requested outcome]

                8. Signature and Contact Information
                [Signature and details]

                9. Certificate of Service
                [Proof of service]

                Analyze the following document and fill in this structure:
                {document}
                Always respond in English, regardless of the document language.""",
                chunk
            )
            response = model.invoke(messages)
            appeal_parts.append(response.content)
        return " ".join(appeal_parts)
    except Exception as e:
        logging.error(f"Error in appeal agent: {e}")
        return f"Error generating appeal: {str(e)}"

async def legal_review_agent(document: str) -> str:
    """Generate a comprehensive legal review following Serbian legal standards."""
    try:
        doc_chunks = chunk_document(document)
        reviews = []
        for chunk in doc_chunks:
            messages = create_messages(
                """You are an expert in Serbian law, an AI legal analyst with deep knowledge of Serbian contract, commercial and civil law.
                Where applicable, follow the guidelines below for specific document types. Create a focused legal review of the document (maximum 750 words). Your goal is to produce a concise, actionable overview that Serbian lawyers can immediately use.

                SUMMARY FOR ENFORCEMENT (3–4 sentences max)
                - Type of document, purpose, and parties
                - Applicable law and jurisdiction
                - Key financial or business obligations
                - Critical compliance status

                HIGH-PRIORITY ANALYSIS
                A. Legal Compliance (Top 3 critical issues)
                - Compliance problems with Serbian law, referencing specific statutes
                - Missing mandatory clauses required by the Serbian Civil Code
                - Consumer protection law issues (if applicable)
                - EU law implications affecting validity

                B. Risk Assessment (Top 3 by severity)
                - Business/legal risks with potential impact
                - Concerns about enforceability before Serbian courts
                - Deviations from Serbian market practice
                - Conflicts with recent Supreme Court precedents

                ACTION PLAN (maximum 5 points)
                - Required amendments for legal compliance
                - Specific clause modifications needed
                - Additional recommended provisions
                - Steps to mitigate risks
                - Practical guidance for implementation

                REVIEW REQUIREMENTS
                - Reference specific Serbian laws, regulations, and case law
                - Focus on essential issues, not formatting
                - Prioritize problems based on legal/business impact
                - Keep the language clear and action-oriented
                - Include business-critical EU law implications (when relevant)

                FINAL SUMMARY
                A 3-sentence conclusion highlighting the most urgent issue requiring immediate attention.

                REVIEW PARAMETERS
                - Each section must be direct and concise
                - Focus on major legal issues, not minor technicalities
                - Include only relevant case law references
                - Maintain practical business context
                - Emphasize any urgent compliance issues

                Analyze the following document according to these parameters:
                {document}
                Always respond in English, regardless of the document language.""",
                chunk
            )
            response = model.invoke(messages)
            reviews.append(response.content)
        return " ".join(reviews)
    except Exception as e:
        logging.error(f"Error in review agent: {e}")
        return f"Error generating review: {str(e)}"

async def legal_lawsuit_agent(document: str) -> str:
    """Generate a formal lawsuit based on the legal document analysis following Serbian legal standards."""
    try:
        doc_chunks = chunk_document(document)
        lawsuit_parts = []
        for chunk in doc_chunks:
            messages = create_messages(
                """You are an AI assistant designed to help Serbian lawyers draft legal complaints and related documents.
                Analyze the document and generate a legal complaint following the structure below:

                [Name of Court]
                [Jurisdiction]
                [Case Number]

                PLAINTIFF: [Extract from document]
                DEFENDANT: [Extract from document]

                COMPLAINT

                I. INTRODUCTION
                [Generate an introduction based on the document]

                II. JURISDICTION AND VENUE
                [Determine the proper jurisdiction]

                III. PARTIES
                [Details about the parties extracted from the document]

                IV. FACTUAL ALLEGATIONS
                [Extract and organize the facts]

                V. CAUSES OF ACTION
                [Legal grounds for the claim]

                VI. DAMAGES
                [Specify the damages]

                VII. PRAYER FOR RELIEF
                [Formulate the requested remedies]

                VIII. REQUEST FOR JUDICIAL PANEL
                [Standard request]

                IX. EXHIBITS
                [List supporting evidence]

                Analyze the following document and complete the structure:
                {document}
                Always respond in English, regardless of the document language.""",
                chunk
            )
            response = model.invoke(messages)
            lawsuit_parts.append(response.content)
        return " ".join(lawsuit_parts)
    except Exception as e:
        logging.error(f"Error in lawsuit agent: {e}")
        return f"Error generating lawsuit: {str(e)}"

async def legal_lawsuit_response_agent(document: str) -> str:
    """Generate a formal response to a lawsuit based on Serbian legal standards."""
    try:
        doc_chunks = chunk_document(document)
        response_parts = []
        for chunk in doc_chunks:
            messages = create_messages(
                """You are an AI assistant designed to help Serbian lawyers prepare legal answers to lawsuits.
                Analyze the document and generate an answer to the complaint using the structure below:

                [Name of Court]
                [Jurisdiction]
                [Case Number]

                Defendant: [Extract from document]
                Address: [Defendant’s Address]
                Phone: [Defendant’s Phone]
                Email: [Defendant’s Email]

                ANSWER TO COMPLAINT

                I. INTRODUCTION
                [Generate an introduction based on the document]

                II. RESPONSE TO FACTUAL ALLEGATIONS
                [Address each allegation made by the plaintiff individually]

                III. LEGAL ARGUMENTS
                [Legal arguments and counterarguments]

                IV. EVIDENCE
                [List and describe supporting evidence]

                V. REQUEST FOR RELIEF
                [Formulate the defendant’s requests]

                VI. EXHIBITS
                [List the exhibits]

                Analyze the following document and complete the structure:
                {document}
                Always respond in English, regardless of the document language.""",
                chunk
            )
            response = model.invoke(messages)
            response_parts.append(response.content)
        return " ".join(response_parts)
    except Exception as e:
        logging.error(f"Error in lawsuit response agent: {e}")
        return f"Error generating lawsuit response: {str(e)}"

async def legal_contract_analysis_agent(document: str) -> str:
    """Analyze legal contracts following Serbian legal standards."""
    try:
        doc_chunks = chunk_document(document)
        analysis_parts = []
        for chunk in doc_chunks:
            messages = create_messages(
                """You are a legal contract analyst specialized in Serbian law.
                Please analyze the following contract according to these criteria:

                1. Basic Elements of the Contract:
                   - Offer and acceptance
                   - Consideration and intent
                   - Capacity to contract
                   - Compliance with the Law on Obligations (Zakon o obligacionim odnosima)

                2. Key Clauses:
                   - Identification and explanation of important provisions
                   - Assessment of clarity and enforceability
                   - Recommendations for improvement
                   - Potential legal ambiguities

                3. Legal Compliance:
                   - Verification of compliance with Serbian laws
                   - References to relevant regulations
                   - Alignment with case law
                   - Regulatory concerns

                4. Risk Assessment:
                   - Legal risks
                   - Financial risks
                   - Operational risks
                   - Recommendations for mitigation

                5. Special Provisions:
                   - Choice of law and jurisdiction
                   - International aspects (if any)
                   - Sector-specific requirements
                   - Data protection and confidentiality

                6. Recommendations for Improvement:
                   - Specific proposed amendments
                   - Additional protective measures
                   - Alignment with best practices
                   - Legal optimization

                Analyze the following contract:
                {document}
                Always respond in English, regardless of the document language.""",
                chunk
            )
            response = model.invoke(messages)
            analysis_parts.append(response.content)
        return " ".join(analysis_parts)
    except Exception as e:
        logging.error(f"Error in contract analysis agent: {e}")
        return f"Error analyzing contract: {str(e)}"

async def legal_chat_helper_agent(document: str, question: str = "") -> str:
    """Interactive chat agent for answering questions about legal documents."""
    try:
        system_message = SystemMessage(content="""
            You are the "Legal Chat Helper Agent," designed to assist users in working with legal documents.
            Your role is to:
            - Guide users through document-related tasks
            - Explain content in simple, easy-to-understand language
            - Help interpret specific sections of documents
            - Suggest relevant actions (summaries, appeals, reviews, etc.)
            - Stay neutral and professional
            - Provide accurate, helpful responses
            
            When responding:
            1. First understand if the user needs:
               - An explanation of the document
               - Help editing or modifying the document
               - Guidance on using other agents
               - General legal questions
                                       
            2. Provide clear, structured guidance
            3. Suggest practical next steps
            4. Base your response strictly on the provided document
            Always respond in English, regardless of the document language.
        """)
        
        human_message = HumanMessage(content=f"""
            Based on this legal document, please help with the following:
            
            User Question: {question}
            
            Document Content:
            ---
            {document}
            ---
            
            Please provide a helpful and detailed response while maintaining professional legal tone.
            Always respond in English, regardless of the document language.
        """)
        
        messages = [system_message, human_message]
        response = model.invoke(messages)
        return response.content
    except Exception as e:
        logging.error(f"Error in chat helper: {e}")
        return "I apologize, but I encountered an error. Could you please rephrase your question or specify what you'd like to know about the document?"
