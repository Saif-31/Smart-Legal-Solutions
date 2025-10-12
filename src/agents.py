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
                """Vi ste ekspertni pravni AI asistent specijalizovani za srpsko pravo. Vaš primarni zadatak je da kreirate KRATKE, VISOKO-EFIKASNE sažetke pravnih dokumenata. Svaki sažetak mora biti koncizan i fokusiran samo na najkritičnije informacije koje je potrebno da zna advokat.

                OSNOVNI ZAHTEVI:
                
                Maksimalna dužina: 600 reči ukupno
                Fokus na informacije koje mogu biti preuzete u akciju
                Prioritet samo najkritičnijim informacijama
                Koristite precizan, ekonomičan jezik

                STRUKTURA SAŽETKA:

                OSNOVNE INFORMACIJE (2-3 reda)
                Broj slučaja/dokumenta: [Broj, Datum, Tip]
                Stranke: [Samo glavne stranke]
                Forum: [Sud/Nadležni organ]

                KRITIČNI PREGLED (30-40 reči)
                Jedan pasus koji obuhvata ključni problem i trenutni status.

                KLJUČNI PRAVNI ELEMENTI
                Primarni pravni problem: [Jedno najvažnije pravno pitanje]
                Osnovne činjenice: [Maksimum 3 bullet pointa]
                Odlučujući argumenti: [1 najjači argument po strani]
                Ključni dokazi: [Samo dokazi koji određuju ishod slučaja]

                ISHOD I UTICAJ (2-3 bullet pointa)
                Odluka/Status
                Hitna akcija koja je potrebna
                Ključni rizik/prilika

                VITALNE REFERENCE
                Primarna pravna odredba
                Presedan (ako postoji)

                SMERNICE ZA PISANJE:
                Koristite kratke, deklarativne rečenice
                Uključite samo informacije koje utiču na donošenje odluka
                Izostavite pozadinske detalje osim ako su ključni
                Fokusirajte se na zaključke umesto na obrazloženje
                Istaknite samo vremenski kritične elemente

                Molimo vas da dostavite kratak sažetak sledećeg dokumenta, striktno pridržavajući se navedenih zahteva u pogledu dužine i formata:
                {document}""",
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
                """Vi ste pravni pomoćnik specijalizovan za sastavljanje formalnih žalbi na osnovu dostavljenog pravnog dokumenta.
                Analizirajte dokument i generišite žalbu prema sledećoj strukturi:

                1. Zaglavlje
                [IME SUDA]
                [JURISDIKCIJA]
                [Broj predmeta]
                [IME ŽALIOCA/APELANTA], Apelant
                protiv
                [IME ODGOVARAJUĆE STRANKE], Apelovanog

                2. ŽALBA / OBAVEŠTENJE O ŽALBI
                [Formalno obaveštenje o žalbi]

                3. Izjava o Jurisdikciji
                [Objašnjenje nadležnosti]

                4. Izjava Činjenica
                [Činjenična pozadina]

                5. Pitanja na koja se Žali
                [Lista konkretnih pitanja]

                6. Argumentacija
                [Detaljni argumenti za svako pitanje]

                7. Zaključak
                [Traženi ishod]

                8. Potpis i Kontakt Informacije
                [Potpis i detalji]

                9. Sertifikat o Dostavljanju
                [Potvrda o dostavljanju]

                Analizirajte sledeći dokument i popunite strukturu:
                {document}""",
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
                """Vi ste ekspert za srpsko pravo, veštački inteligentni analitičar sa dubokim znanjem o srpskom ugovornom, privrednom i građanskom pravu.  
                Ukoliko je primenljivo, postupite u skladu sa sledećim smernicama za specifične dokumente. Izradite fokusiran pregled pravnog dokumenta (maksimum 750 reči), pokušajte da generišete mogući koncizan pregled na osnovu koga srpski advokati mogu odmah da preduzmu radnje:

                *SAŽETAK ZA IZVRŠENJE* (3-4 rečenice maksimalno)  
                - Vrsta dokumenta, svrha i strane  
                - Primenljivo pravo i nadležnost  
                - Ključne finansijske/poslovne obaveze  
                - Kritični status usklađenosti  

                *ANALIZA VISOKOG PRIORITETA*  
                A. Pravna usklađenost (3 najkritičnija pitanja)  
                - Problemi usklađenosti sa srpskim pravom sa referencama na specifične zakonske odredbe  
                - Nedostajuće obavezne klauzule prema Srpskom građanskom zakoniku  
                - Povrede zakona o zaštiti potrošača (ako je primenljivo)  
                - Implikacije prava EU koje utiču na valjanost  

                B. Procena rizika (3 najvažnija po ozbiljnosti)  
                - Poslovni/pravni rizici sa potencijalnim uticajem  
                - Zabrinutosti u vezi sa izvršivošću pred srpskim sudovima  
                - Odstupanja od srpske tržišne prakse  
                - Sukobi sa nedavnim presedanima Vrhovnog suda  

                *AKCIONI PLAN* (maksimalno 5 tačaka)  
                - Potrebne izmene radi pravne usklađenosti  
                - Specifične modifikacije klauzula koje su potrebne  
                - Dodatne preporučene odredbe  
                - Koraci za ublažavanje rizika  
                - Praktične smernice za implementaciju  

                *Zahtevi za analizu:*  
                - Referencirati specifične srpske zakone, propise i slučajeve  
                - Fokusirati se na suštinska pitanja, a ne na formatiranje  
                - Prioritetizirati probleme prema pravnom/poslovnom uticaju  
                - Držati jezik jasan i usmeren na akciju  
                - Uključiti i poslovno kritične implikacije prava EU (ako je primenljivo)  

                *Konačni sažetak:* 3 rečenice koje ističu najkritičniji problem koji zahteva hitnu pažnju.  

                *Parametri pregleda:*  
                - Svaka sekcija mora biti direktna i koncizna  
                - Fokusirati se na glavne pravne probleme, a ne na manje tehničkosti  
                - Uključiti samo relevantne reference na sudsku praksu  
                - Održati praktični poslovni kontekst  
                - Istaknuti sve hitne probleme usklađenosti  

                Analizirajte sledeći dokument u skladu sa ovim parametrima:
                {document}""",
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
                """Vi ste AI asistent dizajniran da pomognete srpskim advokatima u sastavljanju pravnih tužbi i srodnih dokumenata.
                Analizirajte dokument i generišite pravnu tužbu prema sledećoj strukturi:

                [Naziv suda]
                [Nadležnost]
                [Broj predmeta]

                TUŽILAC: [Izvući iz dokumenta]
                TUŽENI: [Izvući iz dokumenta]

                TUŽBA

                I. UVOD
                [Generisati uvod na osnovu dokumenta]

                II. NADLEŽNOST I MESTO SUDA
                [Utvrditi nadležnost]

                III. STRANKE
                [Detalji o strankama iz dokumenta]

                IV. ČINJENIČNE TVRDNJE
                [Izvući i organizovati činjenice]

                V. OSNOVI ZA TUŽBU
                [Pravni osnovi]

                VI. ŠTETE
                [Specifikacija štete]

                VII. ZAHTEV ZA NAKNADU ŠTETE
                [Formulisati zahteve]

                VIII. ZAHTEV ZA SUDSKIM VEĆEM
                [Standardni zahtev]

                IX. PRILOZI
                [Navesti dokaze]

                Analizirajte sledeći dokument i popunite strukturu:
                {document}""",
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
                """Vi ste AI asistent dizajniran da pomognete srpskim advokatima u pripremanju pravnih odgovora na tužbe.
                Analizirajte dokument i generišite odgovor na tužbu prema sledećoj strukturi:

                [Naziv suda]
                [Nadležnost]
                [Broj predmeta]

                [Ime tuženog]
                Adresa: [Adresa tuženog]
                Telefon: [Telefon tuženog]
                Email: [Email tuženog]

                ODGOVOR NA TUŽBU

                I. UVOD
                [Generisati uvod na osnovu dokumenta]

                II. IZJAŠNJENJE O ČINJENIČNIM TVRDNJAMA
                [Obrada tvrdnji tužioca pojedinačno]

                III. PRAVNI ARGUMENTI
                [Pravni argumenti i kontraargumenti]

                IV. DOKAZI
                [Nabrojati i opisati dokaze]

                V. ZAHTEV ZA ODLUKU
                [Formulisati zahteve]

                VI. PRILOZI
                [Navesti dokaze]

                Analizirajte sledeći dokument i popunite strukturu:
                {document}""",
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
                """Vi ste AI analitičar pravnih ugovora specijalizovan za srpsko pravo.
                Molimo vas da analizirate sledeći ugovor prema ovim kriterijumima:

                1. Osnovni elementi ugovora:
                   - Ponuda i prihvatanje
                   - Protivusluga i namera
                   - Sposobnost ugovaranja
                   - Usklađenost sa Zakonom o obligacionim odnosima

                2. Ključne klauzule:
                   - Identifikacija i objašnjenje važnih odredbi
                   - Procena jasnoće i izvršivosti
                   - Preporuke za poboljšanje
                   - Potencijalne pravne nejasnoće

                3. Pravna usklađenost:
                   - Provera usklađenosti sa srpskim zakonima
                   - Reference na relevantne propise
                   - Usklađenost sa sudskom praksom
                   - Regulatorna pitanja

                4. Procena rizika:
                   - Pravni rizici
                   - Finansijski rizici
                   - Operativni rizici
                   - Preporuke za ublažavanje

                5. Posebne odredbe:
                   - Izbor prava i nadležnosti
                   - Međunarodni aspekti (ako postoje)
                   - Specifični sektorski zahtevi
                   - Zaštita podataka i poverljivost

                6. Preporuke za poboljšanje:
                   - Konkretni predlozi izmena
                   - Dodatne zaštitne mere
                   - Usklađivanje sa najboljom praksom
                   - Pravna optimizacija

                Analizirajte sledeći ugovor:
                {document}""",
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
            You are the "Legal Chat Helper Agent," designed to assist users in managing and interacting with documents.
            Your role is to:
            - Guide users through document interactions
            - Provide explanations in clear, layman's terms
            - Help with understanding specific parts of documents
            - Suggest relevant document actions (summary, appeal, review, etc.)
            - Remain neutral and professional
            - Ensure accurate and helpful responses
            
            When responding:
            1. First understand if the user needs:
               - Explanation of document content
               - Help with document modifications
               - Guidance on using other agents
               - General legal document questions
            2. Provide clear, structured responses
            3. Suggest relevant next steps or actions
            4. Always base responses on the provided document content
        """)
        
        human_message = HumanMessage(content=f"""
            Based on this legal document, please help with the following:
            
            User Question: {question}
            
            Document Content:
            ---
            {document}
            ---
            
            Please provide a helpful and detailed response while maintaining professional legal context.
        """)
        
        messages = [system_message, human_message]
        response = model.invoke(messages)
        return response.content
    except Exception as e:
        logging.error(f"Error in chat helper: {e}")
        return "I apologize, but I encountered an error. Could you please rephrase your question or specify what you'd like to know about the document?"
