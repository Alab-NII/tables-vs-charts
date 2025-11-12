# utils
import re


def extract_answer(text):
    for tag in ["<|eot_id|>", "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>"]:
        text = text.replace(tag, "")
    #
    raw_answers = re.findall(r"<ans>\s*(.*?)\s*</ans>", text, re.DOTALL | re.IGNORECASE)
    for ans in raw_answers:
        ans_clean = ans.strip()
        if ans_clean.lower() not in {"your answer", "the answer", "", "insert answer here"}:
            return ans_clean.capitalize()
    #
    # Step 3: Fallback: [Supported], [Refuted], [Neutral]
    bracket_match = re.search(r"\[(Supported|Refuted|Neutral)\]", text, re.IGNORECASE)
    if bracket_match:
        return bracket_match.group(1).capitalize()
    #
    # Step 4: Fallback: "the claim is supported"
    phrase_match = re.search(
        r"\bthe claim (is|appears to be|can be considered to be)?\s*(supported|refuted|neutral)\b",
        text,
        re.IGNORECASE,
    )
    if phrase_match:
        return phrase_match.group(2).capitalize()
    #
    return ""


def format_cells(list_of_lists):
    # Minor -1 for row index (The row in our ground truth index need to be -1. Because the header is row 0.)
    list_of_lists_2 = [[row - 1, col] for row, col in list_of_lists]
    # Convert to list of tuple 
    return [tuple(pair) for pair in list_of_lists_2]


def format_table_as_markdown(table_column_names, table_content_values):
    table_content_values = [[str(element) for element in row] for row in table_content_values]
    # Header
    header = "| " + " | ".join(table_column_names) + " |"
    
    # Rows
    rows = [
        "| " + " | ".join(row) + " |"
        for row in table_content_values
    ]

    # Combine all parts
    markdown_table = "\n".join([header] + rows)
    return markdown_table


def format_table_as_pipe_tag(table_column_names, table_content_values):
    """
    """
    # Convert all elements to strings
    table_content_values = [[str(element) for element in row] for row in table_content_values]
    #
    # Start with the column header line
    # output_lines = ["col : " + " | ".join(table_column_names)]
    output_lines = ["col : " + " | ".join(str(col) for col in table_column_names)]
    #
    # Add each row with row number
    for idx, row in enumerate(table_content_values, 1):
        output_lines.append(f"row {idx} : " + " | ".join(row))
    #
    # Join all lines into a single string
    return "\n".join(output_lines)


def format_caption_and_table(table_column_names, table_content_values, caption, type_="pipe_tagging"):
    if type_ == "markdown":
        table = format_table_as_markdown(table_column_names, table_content_values)
    if type_ == "pipe_tagging":
        table = format_table_as_pipe_tag(table_column_names, table_content_values)
    return f"Caption: {caption}\n\nTable Data:\n{table}\n"


def format_table(table_column_names, table_content_values, type_="pipe_tagging"):
    if type_ == "markdown":
        table = format_table_as_markdown(table_column_names, table_content_values)
    if type_ == "pipe_tagging":
        table = format_table_as_pipe_tag(table_column_names, table_content_values)
    return f"{table}\n" # remove Table Data because in the prompt already have