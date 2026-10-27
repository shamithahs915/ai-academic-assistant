import os
import json
import re
from typing import List, Dict, Any, Optional
from config import MARK_SCHEMES, GEMINI_MODELS, calibrate_scheme_components
from diagram_engine import DiagramEngine
from structured_output import StructuredOutputPipeline

NOT_ENOUGH_INFO_MSG = "The uploaded PDF does not contain enough relevant information to answer this question."

STOP_WORDS = {
    "what", "is", "explain", "describe", "discuss", "with", "neat", "diagram",
    "and", "the", "for", "from", "concept", "write", "short", "note", "notes",
    "define", "how", "does", "give", "brief", "state", "list", "illustrate"
}

class AcademicAnswerGenerator:
    """
    Intelligent Answer and Marking Scheme Generator.
    Produces exam-ready, mark-calibrated answers with transparent scoring rubrics
    and labeled architectural/workflow diagrams, strictly grounded in uploaded PDF documents.
    Transforms raw extracted text into structured, typed, user-friendly responses.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name

    def generate_scheme_and_answer(self, question: str, marks: int, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main entry point for generating mark-appropriate answer, marking scheme rubric,
        and conceptual diagram strictly grounded in the provided context_chunks.
        """
        # If no context chunks were retrieved, immediately indicate lack of information
        if not context_chunks:
            return {
                "question": question,
                "marks": marks,
                "scheme": {
                    "target_marks": marks,
                    "components": [
                        {"component": "Relevance & Content Verification", "marks": marks}
                    ],
                    "total_allocated": marks
                },
                "answer_markdown": NOT_ENOUGH_INFO_MSG,
                "structured_output": {
                    "schema_version": "2.0",
                    "topic": question,
                    "content_type": "conceptual_definition",
                    "domain": "Academic Repository",
                    "executive_summary": NOT_ENOUGH_INFO_MSG,
                    "key_entities": [],
                    "sections": [],
                    "technical_attributes": {},
                    "missing_or_uncertain_fields": ["All required content is absent from the selected document."],
                    "examiner_takeaways": ["Verify that your question is covered in the active document."]
                },
                "is_structured": False,
                "diagram_code": "",
                "diagram_needed": False,
                "diagram_title": "",
                "diagram_message": "No diagram generated because relevant content was not found in the uploaded document.",
                "tips_for_full_marks": [
                    "Ensure the question is directly covered in the uploaded course materials."
                ],
                "sources": [],
                "has_sufficient_context": False
            }

        scheme_template = MARK_SCHEMES.get(marks, MARK_SCHEMES[10])
        diagram_needed = DiagramEngine.should_include_diagram(question, marks, context_chunks=context_chunks)

        # Context compilation from RAG chunks
        context_text = "\n\n".join([
            f"[Source: {c.get('source_doc', 'Document')} | Page {c.get('page', 1)}]\n{c['text']}"
            for c in context_chunks
        ])

        # Attempt Gemini API Generation if key exists
        if self.api_key:
            try:
                result = self._generate_with_gemini(question, marks, context_text, scheme_template, diagram_needed)
                if result:
                    result["sources"] = context_chunks
                    return result
            except Exception as e:
                print(f"Gemini API call failed, falling back to academic engine: {e}")

        # Grounded Academic Knowledge Synthesis Engine
        return self._generate_with_synthesis_engine(question, marks, context_chunks, scheme_template, diagram_needed)

    def _generate_with_gemini(self, question: str, marks: int, context_text: str, 
                              scheme_template: Dict, diagram_needed: bool) -> Optional[Dict[str, Any]]:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)

        system_prompt = f"""
You are an expert Academic Professor and Chief Paper Evaluator in Computer Science and Engineering at Siddaganga Institute of Technology (SIT) / VTU.
Your task is to transform academic source content into a validated, highly structured academic response strictly tailored to {marks} Marks.

CRITICAL INSTRUCTIONS:
1. Treat the provided CONTEXT as raw source data to analyze, NOT as the final response.
2. Analyze the context to extract key entities, definitions, structural components, operational steps, and technical attributes.
3. If the context does not contain enough relevant information to answer the question, return:
   "answer_markdown": "{NOT_ENOUGH_INFO_MSG}"
   "diagram_code": ""
   "has_sufficient_context": false
4. You MUST return a JSON object strictly conforming to this schema:
{{
    "schema_version": "2.0",
    "topic": "{question}",
    "content_type": "conceptual_definition" | "process_workflow" | "architectural_system" | "comparative_analysis",
    "domain": "Domain category (e.g. Operating Systems / Computer Networks)",
    "executive_summary": "Crisp 1-2 sentence core conceptual summary",
    "key_entities": ["Entity1", "Entity2", "Entity3"],
    "sections": [
        {{
            "heading": "Section Heading",
            "type": "key_value" | "table" | "ordered_steps" | "bullet_points",
            "items": [{{"key": "Key Name", "value": "Value description"}}],
            "table_data": {{"headers": ["Col1", "Col2"], "rows": [["Val1", "Val2"]]}},
            "steps": [{{"step_number": 1, "phase_name": "Phase Name", "action": "Action details", "resulting_state": "Result state"}}],
            "points": ["Structured point 1", "Structured point 2"],
            "summary": "Brief section overview"
        }}
    ],
    "scheme": {{
        "target_marks": {marks},
        "components": [
            {{"component": "Rubric section name", "marks": 1.0}}
        ],
        "total_allocated": {marks}
    }},
    "diagram_code": "Valid Mermaid.js code or empty string",
    "diagram_title": "Title for diagram or empty string",
    "tips_for_full_marks": [
        "Examiner scoring takeaway 1"
    ]
}}

MARKS-BASED POINTS REQUIREMENT:
- Required number of points = {marks} × 2 = {marks * 2} points  
- Generate EXACTLY {marks * 2} distinct, meaningful, non-repetitive, factually accurate numbered points
- Number every point in the answer (1. 2. 3. ...)
- Do NOT fabricate information to reach the count — if context supports fewer facts, generate fewer but accurate points
- Verify your point count equals {marks * 2} before returning the JSON
- Add a top-level field: "generated_points_count": <actual count of points you generated>

QUESTION TYPE HANDLING:
- Define: Give the definition first as Point 1, followed by relevant operational details.
- Explain: Provide distinct points with clear explanations.
- Compare: Use a comparison table or comparative points.
- List: Provide the requested items directly and concisely.
- Discuss: Include relevant arguments, explanations, and a concluding evaluation.
- Numerical problems: Show formulas, calculations, working steps, and the final answer without adding irrelevant points.

FORMAT the "answer_markdown" field as:
**Question:** {question}

**Marks:** {marks}

**Answer:**

1. [First relevant point]
2. [Second relevant point]
... [Continue until {marks * 2} points are reached]

📌 **Points Generated:** <count> / {marks * 2} required ({marks} marks × 2)

Ensure the sum of component marks exactly equals {marks}. Return ONLY the JSON object.

CONTEXT FROM ACADEMIC MATERIALS:
\"\"\"
{context_text}
\"\"\"

QUESTION: {question}
"""

        response = model.generate_content(
            system_prompt,
            generation_config={"response_mime_type": "application/json"}
        )

        try:
            data = json.loads(response.text)
            
            # Check for lack of information
            if data.get("answer_markdown") == NOT_ENOUGH_INFO_MSG:
                return {
                    "question": question,
                    "marks": marks,
                    "scheme": {
                        "target_marks": marks,
                        "components": [{"component": "Relevance Verification", "marks": marks}],
                        "total_allocated": marks
                    },
                    "answer_markdown": NOT_ENOUGH_INFO_MSG,
                    "structured_output": data,
                    "is_structured": False,
                    "diagram_code": "",
                    "diagram_needed": False,
                    "diagram_title": "",
                    "tips_for_full_marks": [],
                    "has_sufficient_context": False
                }

            diagram_code = data.get("diagram_code", "").strip()
            
            # Validate diagram syntax if code was generated
            if diagram_code:
                is_valid, err_msg = DiagramEngine.validate_mermaid_syntax(diagram_code)
                if not is_valid:
                    # Retry once to fix syntax error
                    try:
                        retry_prompt = f"The previous Mermaid diagram had syntax errors: {err_msg}.\nOriginal Code:\n{diagram_code}\nPlease fix the Mermaid syntax. Return ONLY valid Mermaid code, or an empty string if context cannot support it."
                        retry_res = model.generate_content(retry_prompt)
                        cleaned_retry = DiagramEngine.sanitize_mermaid_code(retry_res.text)
                        is_valid2, _ = DiagramEngine.validate_mermaid_syntax(cleaned_retry)
                        if is_valid2:
                            data["diagram_code"] = cleaned_retry
                        else:
                            data["diagram_code"] = ""
                            data["diagram_message"] = "Diagram could not be reliably validated against PDF context."
                    except Exception:
                        data["diagram_code"] = ""
                        data["diagram_message"] = "Diagram could not be reliably validated against PDF context."
                else:
                    data["diagram_code"] = DiagramEngine.sanitize_mermaid_code(diagram_code)
            else:
                data["diagram_code"] = ""

            data["diagram_needed"] = bool(data.get("diagram_code"))
            data["marks"] = marks
            data["question"] = question

            # Validate and record points count
            generated_count = data.get("generated_points_count", 0)
            target_points = marks * 2
            data["target_points"] = target_points
            data["generated_points_count"] = generated_count
            
            # Calibrate scheme components
            if "scheme" in data and "components" in data["scheme"]:
                data["scheme"]["components"] = calibrate_scheme_components(data["scheme"]["components"], marks)
                data["scheme"]["total_allocated"] = marks
                data["scheme"]["target_marks"] = marks

            # Validate and compile structured output
            is_valid, val_errors = StructuredOutputPipeline.validate_structured_output(data)
            if is_valid:
                # Keep Gemini's custom marks-formatted answer_markdown if available
                existing_ans = data.get("answer_markdown", "")
                if not existing_ans or existing_ans == NOT_ENOUGH_INFO_MSG or "Answer:" not in existing_ans:
                    data["answer_markdown"] = StructuredOutputPipeline.render_to_markdown(data, marks)
                data["structured_output"] = data
                data["is_structured"] = True
                data["has_sufficient_context"] = True
            else:
                print(f"Gemini structured output validation warnings: {val_errors}")
                # Fallback to local structured pipeline
                return None

            return data
        except Exception as err:
            print(f"Error parsing Gemini response JSON: {err}")
            return None

    def _extract_grounded_points(self, question: str, marks: int, usable_sentences: List[str], context_chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Extracts up to marks * 2 distinct, meaningful, grounded points adapted to question type.
        Prioritizes definition for define questions, comparative points for compare questions,
        and ensures no duplicate, boilerplate header, or fabricated statements.
        """
        target_points = marks * 2
        q_lower = question.lower()
        q_keywords = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', q_lower) if w not in STOP_WORDS]

        # Identify Question Type
        if any(w in q_lower for w in ["calculate", "compute", "solve", "numerical", "find"]):
            q_type = "numerical"
        elif any(w in q_lower for w in ["define", "what is", "meaning of", "concept of"]):
            q_type = "define"
        elif any(w in q_lower for w in ["compare", "difference", "distinguish", "contrast", "vs", "versus"]):
            q_type = "compare"
        elif any(w in q_lower for w in ["list", "state", "enumerate", "name the", "types of"]):
            q_type = "list"
        elif any(w in q_lower for w in ["discuss", "evaluate", "critique"]):
            q_type = "discuss"
        else:
            q_type = "explain"

        institutional_markers = [
            "siddaganga institute of technology",
            "department of computer science",
            "course:",
            "course code",
            "autonomous institution",
            "visvesvaraya",
            "academic year",
            "dr. shivakumara",
            "affiliated to vtu"
        ]

        # 1. Collect candidate statements from context sentences and chunks
        raw_candidates = []
        
        # Helper to clean candidate
        def sanitize_candidate(text: str) -> Optional[str]:
            c = text.strip()
            # Remove leading institutional boilerplate if any
            for im in institutional_markers:
                if im in c.lower() and len(c) < 140:
                    return None
            # Strip module headers and section titles like "MODULE 2: ... 2.1 ..."
            c = re.sub(r'(?i)^(?:module|unit|chapter)\s*\d+:[^\n]*\n?', '', c).strip()
            c = re.sub(r'^\d+\.\d+\s+[A-Za-z\s]+(?:\n|$)', '', c).strip()
            # Strip bullets
            c = re.sub(r'^[•\-\*\d+\.]\s*', '', c).strip()
            if len(c) < 18:
                return None
            if any(im in c.lower() for im in ["siddaganga institute", "visvesvaraya technological", "department of computer"]):
                return None
            return c

        for s in usable_sentences:
            s_clean = s.strip()
            # If compound sentence with semicolon, split into meaningful clauses
            if ";" in s_clean:
                clauses = [c.strip() for c in s_clean.split(";") if len(c.strip()) > 18]
                if len(clauses) > 1:
                    for cl in clauses:
                        san = sanitize_candidate(cl)
                        if san:
                            raw_candidates.append(san)
                    continue
            san = sanitize_candidate(s_clean)
            if san:
                raw_candidates.append(san)

        # 2. If needed, pull additional distinct statements from bullet lines in chunks
        if len(raw_candidates) < target_points:
            for c in context_chunks:
                for line in c.get("text", "").split("\n"):
                    san = sanitize_candidate(line)
                    if san and san not in raw_candidates:
                        raw_candidates.append(san)

        # 3. Question-type specific ordering
        ordered_candidates = []
        if q_type == "define":
            # Score and rank candidates by matching question keywords
            def def_score(cand: str) -> int:
                c_low = cand.lower()
                kw_score = sum(2 for k in q_keywords if re.search(r'\b' + re.escape(k) + r'\b', c_low))
                verb_score = 1 if any(w in c_low for w in ["is a", "is defined as", "refers to", "scheme that", "capability of"]) else 0
                return kw_score + verb_score

            def_candidates = [c for c in raw_candidates if def_score(c) > 0]
            def_candidates.sort(key=def_score, reverse=True)
            other_candidates = [c for c in raw_candidates if c not in def_candidates]
            ordered_candidates = def_candidates + other_candidates
        elif q_type == "compare":
            # Prioritize comparative statements
            comp_candidates = [
                c for c in raw_candidates 
                if any(w in c.lower() for w in ["whereas", "while", "unlike", "differs", "contrast", "instead of", "compared to"])
            ]
            other_candidates = [c for c in raw_candidates if c not in comp_candidates]
            ordered_candidates = comp_candidates + other_candidates
        elif q_type == "list":
            ordered_candidates = sorted(raw_candidates, key=lambda x: len(x))
        else:
            ordered_candidates = raw_candidates

        # 4. Deduplicate and clean
        final_points = []
        seen = set()
        for c in ordered_candidates:
            clean = c.strip().rstrip(".")
            norm = re.sub(r'[^a-zA-Z0-9]', '', clean.lower())
            if len(clean) > 18 and norm not in seen:
                seen.add(norm)
                final_points.append(clean)
            if len(final_points) >= target_points:
                break

        return final_points

    def _generate_with_synthesis_engine(self, question: str, marks: int, context_chunks: List[Dict[str, Any]], 
                                       scheme_template: Dict, diagram_needed: bool) -> Dict[str, Any]:
        """
        Academic synthesis engine that formulates structured answers strictly from document context
        without inventing statements, generic fillers, or unsupported diagrams.
        """
        combined_text = " ".join([c["text"] for c in context_chunks])
        
        # Extract keywords from question to check relevance
        GENERIC_ACADEMIC_TERMS = {'algorithm', 'system', 'process', 'approach', 'method', 'concept', 'structure', 'scheme', 'technique', 'types', 'type', 'functions', 'function'}
        q_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())) - STOP_WORDS
        text_lower = combined_text.lower()
        
        # Word boundary matching for precision
        matching_keywords = [k for k in q_tokens if re.search(r'\b' + re.escape(k) + r'\b', text_lower)]
        non_generic_tokens = q_tokens - GENERIC_ACADEMIC_TERMS

        # Require that specific, non-generic subject keywords are present in context
        if non_generic_tokens:
            matching_specific = [k for k in non_generic_tokens if re.search(r'\b' + re.escape(k) + r'\b', text_lower)]
            is_relevant_context = (len(matching_specific) / len(non_generic_tokens)) >= 0.50
        else:
            is_relevant_context = len(matching_keywords) > 0
        
        # Split sentences from context
        sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', combined_text) if len(s.strip()) > 15]
        
        # Filter relevant sentences containing question keywords
        relevant_sentences = []
        for s in sentences:
            s_lower = s.lower()
            if any(k in s_lower for k in matching_keywords):
                relevant_sentences.append(s)

        usable_sentences = relevant_sentences if relevant_sentences else [s for s in sentences if len(s) > 25]

        # If insufficient topical coverage or no usable academic text, state insufficient information
        if not is_relevant_context or not usable_sentences:
            return {
                "question": question,
                "marks": marks,
                "scheme": {
                    "target_marks": marks,
                    "components": [
                        {"component": "Relevance & Content Verification", "marks": marks}
                    ],
                    "total_allocated": marks
                },
                "answer_markdown": NOT_ENOUGH_INFO_MSG,
                "structured_output": {
                    "schema_version": "2.0",
                    "topic": question,
                    "content_type": "conceptual_definition",
                    "domain": "Academic Repository",
                    "executive_summary": NOT_ENOUGH_INFO_MSG,
                    "key_entities": [],
                    "sections": [],
                    "technical_attributes": {},
                    "missing_or_uncertain_fields": ["All required content is absent from the selected document."],
                    "examiner_takeaways": ["Verify that your question is covered in the active document."]
                },
                "is_structured": False,
                "diagram_code": "",
                "diagram_needed": False,
                "diagram_title": "",
                "diagram_message": "No diagram generated because relevant content was not found in the uploaded document.",
                "tips_for_full_marks": [
                    "Ensure the question is directly covered in the uploaded course materials."
                ],
                "sources": context_chunks,
                "has_sufficient_context": False
            }

        # Attempt to synthesize grounded diagram from context
        diagram_code = ""
        diagram_title = ""
        diagram_msg = None
        if diagram_needed:
            code, title, msg = DiagramEngine.generate_diagram_from_context(question, context_chunks)
            if code:
                diagram_code = code
                diagram_title = title
            else:
                diagram_msg = msg

        # Calculate target points
        target_points = marks * 2

        # Execute Structured Output Pipeline: Content Analysis -> Schema Selection -> Transformation
        analysis = StructuredOutputPipeline.analyze_content(question, context_chunks)
        structured_data = StructuredOutputPipeline.generate_structured_data(
            question=question,
            marks=marks,
            context_chunks=context_chunks,
            analysis=analysis,
            target_points=target_points
        )

        # Validate generated structure
        is_valid, validation_errors = StructuredOutputPipeline.validate_structured_output(structured_data)
        if not is_valid:
            print(f"Structured output validation errors: {validation_errors}")

        # Build marks-based numbered answer from usable sentences respecting question type
        numbered_points = self._extract_grounded_points(question, marks, usable_sentences, context_chunks)
        generated_count = len(numbered_points)

        # Update structured data point counts
        structured_data["required_points"] = target_points
        structured_data["generated_points"] = generated_count

        # Build answer_markdown in required format
        answer_lines = [
            f"**Question:** {question}\n",
            f"**Marks:** {marks}\n",
            "**Answer:**\n"
        ]
        for i, pt in enumerate(numbered_points, 1):
            answer_lines.append(f"{i}. {pt}.")
        answer_lines.append("")
        answer_lines.append(f"📌 **Points Generated:** {generated_count} / {target_points} required ({marks} marks × 2)")
        full_answer = "\n".join(answer_lines)

        # Determine official mark components
        if marks == 2:
            components = [
                {"component": "Formal Definition / Core Technical Concept", "marks": 1.0},
                {"component": "Key Characteristic / Operational Principle", "marks": 1.0}
            ]
        elif marks == 5:
            components = [
                {"component": "Introduction & Formal Definition", "marks": 1.0},
                {"component": "Working Mechanism & Core Operational Points", "marks": 2.5},
                {"component": "Technical Details & Context", "marks": 1.5}
            ]
        elif marks == 10:
            components = [
                {"component": "Conceptual Introduction & Theoretical Background", "marks": 1.5},
                {"component": "Neat Labeled Architecture / Flow Diagram", "marks": 3.0},
                {"component": "Detailed Technical Breakdown & Step-by-Step Working", "marks": 3.5},
                {"component": "Operational Rules & System Analysis", "marks": 2.0}
            ]
        else: # 15 marks
            components = [
                {"component": "In-depth Theoretical Foundation & System Principles", "marks": 2.0},
                {"component": "Exhaustive System Architecture & Labeled Diagram", "marks": 4.0},
                {"component": "Detailed Multi-Stage Component & Protocol Analysis", "marks": 5.0},
                {"component": "Operational Workflow & Technical Details", "marks": 2.5},
                {"component": "Comparative Analysis, Trade-offs & Concluding Remarks", "marks": 1.5}
            ]

        components = calibrate_scheme_components(components, marks)

        return {
            "question": question,
            "marks": marks,
            "scheme": {
                "target_marks": marks,
                "components": components,
                "total_allocated": marks
            },
            "structured_output": structured_data,
            "is_structured": True,
            "answer_markdown": full_answer,
            "diagram_code": diagram_code,
            "diagram_needed": bool(diagram_code),
            "diagram_title": diagram_title,
            "diagram_message": diagram_msg,
            "target_points": target_points,
            "generated_points_count": generated_count,
            "tips_for_full_marks": [
                f"Structure your response strictly following the {marks}-mark rubric.",
                "Highlight key terminology in bold during written examinations.",
                f"Examiners allot {components[1]['marks'] if len(components) > 1 else 1} marks specifically for neat schematics where applicable."
            ],
            "sources": context_chunks,
            "has_sufficient_context": True
        }
