import streamlit as st
import mysql.connector
import magic
import io
import os
import tempfile
import base64

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT

# Register custom font
pdfmetrics.registerFont(TTFont('helvetica', 'helvetica.ttf'))
base_font_name = 'helvetica'

# Streamlit UI
st.title("HR's Requirement")
with st.form("HR form"):
    position = st.text_input(label="Position", placeholder="Please enter the HR's required position")
    exp = st.text_input(label="Minimum Experience (in years)", placeholder="Please enter the minimum experience required (in years)")
    submitted = st.form_submit_button("Submit")

# MySQL connection
mydb = mysql.connector.connect(
    user='root', password='password',
    host='127.0.0.1', database='resumes_database',
    auth_plugin='mysql_native_password'
)
cur = mydb.cursor()

# Resume fetch function
def fetch_resumes(cur, position, exp):
    query = """
        SELECT E.NAME, E.EMAIL, E.LOCATION, E.SCORE, E.RESUME
        FROM EMPLOYEES E
        INNER JOIN RESUME_DETAILS RD ON E.EMAIL = RD.EMAIL
        WHERE RD.predicted_position = %s 
          AND (RD.EXPERIENCE >= %s OR RD.EXPERIENCE IS NULL OR RD.EXPERIENCE = 0)
        ORDER BY E.SCORE DESC
    """
    cur.execute(query, (position, exp))
    return cur.fetchall()

# Check if data is valid PDF
def is_valid_pdf(data: bytes) -> bool:
    return data.startswith(b'%PDF')

# Convert DOCX to PDF using ReportLab
def docx_to_pdf_reportlab(docx_data: bytes) -> bytes:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_docx:
            temp_docx.write(docx_data)
            docx_path = temp_docx.name

        doc = Document(docx_path)
        pdf_buffer = io.BytesIO()
        doc_template = SimpleDocTemplate(pdf_buffer, pagesize=(8.5 * inch, 11 * inch))
        styles = getSampleStyleSheet()
        story = []

        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            if paragraph.style.name == 'Title':
                story.append(Paragraph(text, styles['Title']))
            elif paragraph.style.name == 'Heading 1':
                story.append(Paragraph(text, styles['Heading1']))
            elif paragraph.style.name == 'Heading 2':
                story.append(Paragraph(text, styles['Heading2']))
            elif 'Bold' in paragraph.style.name:
                story.append(Paragraph(f"<font name='{base_font_name}'><b>{text}</b></font>", styles['Normal']))
            elif 'Italic' in paragraph.style.name:
                story.append(Paragraph(f"<font name='{base_font_name}'><i>{text}</i></font>", styles['Normal']))
            else:
                story.append(Paragraph(f"<font name='{base_font_name}'>{text}</font>", styles['Normal']))
            story.append(Spacer(1, 0.2 * inch))

        doc_template.build(story)
        pdf_data = pdf_buffer.getvalue()
        os.remove(docx_path)
        return pdf_data

    except Exception as e:
        st.error(f"Error converting DOCX to PDF: {e}")
        return None

# If form submitted, fetch and display resumes
if submitted:
    resumes = fetch_resumes(cur, position, exp)

    if resumes:
        colms = st.columns((1, 2, 2, 1))
        headers = ["Name", "Email", "Location", "Score"]
        for col, header in zip(colms, headers):
            col.write(f"**{header}**")

        for name, email, location, score, resume_data in resumes:
            col1, col2, col3, col4 = st.columns((1, 2, 2, 1))
            col1.write(name)
            col2.write(email)
            col3.write(location)
            col4.write(score)

            if resume_data:
                file_type = magic.from_buffer(resume_data, mime=True)

                if is_valid_pdf(resume_data):
                    base64_pdf = base64.b64encode(resume_data).decode('utf-8')
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="1200" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)

                elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                    pdf_data = docx_to_pdf_reportlab(resume_data)
                    if pdf_data:
                        pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
                        st.markdown(
                            f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="800" height="1200" type="application/pdf"></iframe>',
                            unsafe_allow_html=True
                        )
                    else:
                        base64_docx = base64.b64encode(resume_data).decode('utf-8')
                        docx_display = (
                            f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{base64_docx}" '
                            f'download="{name}_resume.docx">Download {name}\'s Resume (DOCX)</a>'
                        )
                        st.markdown(docx_display, unsafe_allow_html=True)
                else:
                    st.error(f"Unsupported file type for {name}: {file_type}")
            else:
                st.info(f"No resume uploaded by {name}.")
    else:
        st.warning("No matching resumes found.")
