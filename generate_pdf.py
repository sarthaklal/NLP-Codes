"""
===============================================================================
  Generate PDF Report for NLP Unit 5 Practice Programs
===============================================================================
  Runs all 4 programs, captures output, and generates a combined PDF
  with both source code and results.
===============================================================================
"""

import subprocess
import sys
import os
from datetime import datetime

def capture_output(script_path):
    """Run a script and capture its output."""
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, cwd=os.path.dirname(script_path)
    )
    return result.stdout + (result.stderr if result.returncode != 0 else "")


def read_source(script_path):
    """Read source code of a script."""
    with open(script_path, "r") as f:
        return f.read()


def generate_pdf():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    programs = [
        ("1. CHATBOT", os.path.join(base_dir, "1_chatbot.py")),
        ("2. QUESTION ANSWERING", os.path.join(base_dir, "2_question_answering.py")),
        ("3. TEXT SUMMARIZATION", os.path.join(base_dir, "3_summarization.py")),
        ("4. MACHINE TRANSLATION", os.path.join(base_dir, "4_machine_translation.py")),
    ]

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, mm
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, PageBreak,
        Table, TableStyle, Preformatted
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    pdf_path = os.path.join(base_dir, "NLP_Unit5_Practice_Programs.pdf")

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Title'],
        fontSize=22, spaceAfter=6, textColor=HexColor('#1a237e'),
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        'SubTitle', parent=styles['Normal'],
        fontSize=12, spaceAfter=20, textColor=HexColor('#424242'),
        alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        'CustomHeading', parent=styles['Heading1'],
        fontSize=16, spaceAfter=10, spaceBefore=15,
        textColor=HexColor('#0d47a1'),
    )
    subheading_style = ParagraphStyle(
        'CustomSubHeading', parent=styles['Heading2'],
        fontSize=13, spaceAfter=8, spaceBefore=10,
        textColor=HexColor('#1565c0'),
    )
    code_style = ParagraphStyle(
        'CodeStyle', parent=styles['Code'],
        fontSize=6.5, leading=8, fontName='Courier',
        leftIndent=10, rightIndent=10,
        backColor=HexColor('#f5f5f5'),
        borderColor=HexColor('#e0e0e0'),
        borderWidth=0.5, borderPadding=5,
    )
    output_style = ParagraphStyle(
        'OutputStyle', parent=styles['Code'],
        fontSize=7, leading=9, fontName='Courier',
        leftIndent=10, rightIndent=10,
        backColor=HexColor('#e8f5e9'),
        borderColor=HexColor('#a5d6a7'),
        borderWidth=0.5, borderPadding=5,
    )

    elements = []

    # Title Page
    elements.append(Spacer(1, 2*inch))
    elements.append(Paragraph("NLP Unit 5", title_style))
    elements.append(Paragraph("Practice Programs", title_style))
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph(
        "Chatbot • Question Answering • Summarization • Machine Translation",
        subtitle_style
    ))
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        subtitle_style
    ))
    elements.append(Paragraph("Implementation and Results", subtitle_style))
    elements.append(PageBreak())

    # Table of Contents
    elements.append(Paragraph("Table of Contents", heading_style))
    elements.append(Spacer(1, 0.2*inch))
    toc_items = [
        "1. Chatbot — Rule-Based + TF-IDF Retrieval",
        "2. Question Answering — TF-IDF Extractive QA",
        "3. Text Summarization — TF-IDF + TextRank",
        "4. Machine Translation — IBM Model 1 (Statistical)",
    ]
    for item in toc_items:
        elements.append(Paragraph(item, styles['Normal']))
        elements.append(Spacer(1, 4))
    elements.append(PageBreak())

    # Process each program
    for prog_name, prog_path in programs:
        print(f"Processing {prog_name}...")

        # Section heading
        elements.append(Paragraph(f"Program {prog_name}", heading_style))
        elements.append(Spacer(1, 0.1*inch))

        # Source code section
        elements.append(Paragraph("Source Code:", subheading_style))
        source = read_source(prog_path)
        # Escape XML special chars for ReportLab
        source_escaped = (source
                         .replace("&", "&amp;")
                         .replace("<", "&lt;")
                         .replace(">", "&gt;"))
        elements.append(Preformatted(source_escaped, code_style))
        elements.append(Spacer(1, 0.2*inch))

        # Output section
        elements.append(Paragraph("Program Output / Results:", subheading_style))
        print(f"  Running {prog_path}...")
        output = capture_output(prog_path)
        if not output.strip():
            output = "[No output captured - check dependencies]"
        output_escaped = (output
                         .replace("&", "&amp;")
                         .replace("<", "&lt;")
                         .replace(">", "&gt;"))
        elements.append(Preformatted(output_escaped, output_style))
        elements.append(PageBreak())

    # Build PDF
    doc.build(elements)
    print(f"\n✅ PDF generated successfully: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    generate_pdf()
