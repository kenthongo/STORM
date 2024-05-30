from typing import List, TypedDict
from modules.generate_sections import WikiSection
from modules.generate_initial_outline import Outline
from modules.generate_perspectives import Editor
from .interview_state import InterviewState


class ResearchState(TypedDict):
    topic: str
    outline: Outline
    editors: List[Editor]
    interview_results: List[InterviewState]
    # The final sections output
    sections: List[WikiSection]
    article: str