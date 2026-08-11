from langchain.tools import tool

@tool
def calculate_cgpa(sgpas: list[float]) -> str:
    """
    Calculate the cumulative GPA from semester GPAs.

    Args:
        sgpas: List of semester GPAs.

    Returns:
        A string containing the calculated CGPA.
    """

    if not sgpas:
        return "Please provide at least one semester GPA."

    if any(gpa < 0 or gpa > 10 for gpa in sgpas):
        return "Each semester GPA must be between 0 and 10."

    cgpa = sum(sgpas) / len(sgpas)

    return f"Your CGPA is {cgpa:.2f}"