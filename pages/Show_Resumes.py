import streamlit as st
import mysql.connector
from typing import List, Tuple
import magic
import io
import base64
import tempfile
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from docx import Document
from docx.opc.exceptions import PackageNotFoundError

st.set_page_config(page_title="Show Resumes")

font_path = "Helvetica.ttf"
try:
    with open(font_path, "rb"):
        pass
except FileNotFoundError:
    font_path = None

if font_path:
    pdfmetrics.registerFont(TTFont('Helvetica', font_path))
    base_font_name = 'helvetica'
else:
    base_font_name = 'helvetica'


def connect_to_database() -> mysql.connector.MySQLConnection:
    try:
        mydb = mysql.connector.connect(
            user='root',
            password='password',
            host='127.0.0.1',
            database='resumes_database',
            auth_plugin='mysql_native_password'
        )
        return mydb
    except mysql.connector.Error as e:
        st.error(f"Database connection error: {e}")
        return None


def get_categories_from_db(mydb: mysql.connector.MySQLConnection) -> List[str]:
    if not mydb:
        return []
    try:
        cursor = mydb.cursor()
        cursor.execute("SELECT DISTINCT category FROM employees")
        categories = [row[0] for row in cursor.fetchall()]
        return categories
    except mysql.connector.Error as e:
        st.error(f"Database error: {e}")
        return []


def get_resumes_from_db(mydb: mysql.connector.MySQLConnection, category: str) -> List[Tuple]:
    if not mydb:
        return []
    try:
        cursor = mydb.cursor()
        query = f"""
            SELECT  
                Name, 
                Email, 
                location, 
                Resume,
                score,
                category
            FROM employees 
            WHERE LOWER(category) = '{category.lower()}'
            ORDER BY score DESC
        """
        cursor.execute(query)
        resumes = cursor.fetchall()
        return resumes
    except mysql.connector.Error as e:
        st.error(f"Error fetching resumes: {e}")
        return []


def is_valid_pdf(data: bytes) -> bool:
    return data.startswith(b'%PDF')


def docx_to_pdf_reportlab(docx_data: bytes) -> bytes:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_docx:
            temp_docx.write(docx_data)
            docx_path = temp_docx.name

        try:
            doc = Document(docx_path)
        except PackageNotFoundError as e:
            st.error(f"Error: Invalid DOCX file: {e}")
            os.remove(docx_path)
            return None

        pdf_buffer = io.BytesIO()
        doc_template = SimpleDocTemplate(pdf_buffer, pagesize=(8.5*inch, 11*inch))
        styles = getSampleStyleSheet()
        story = []

        for paragraph in doc.paragraphs:
            text = paragraph.text
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

        if hasattr(doc, 'relationships'):
            for rel in doc.relationships:
                if rel.reltype == "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image":
                    try:
                        img_path = doc.get_part(rel.target_partname).blob
                        with tempfile.NamedTemporaryFile(delete=False) as temp_img:
                            temp_img.write(img_path)
                            img_path_name = temp_img.name
                        img = Image(img_path_name)
                        img.drawWidth = 4 * inch
                        img.drawHeight = 3 * inch
                        story.append(img)
                        story.append(Spacer(1, 0.2 * inch))
                        os.remove(img_path_name)

                    except Exception as img_err:
                        st.error(f"Error adding image to PDF: {img_err}")

        doc_template.build(story)
        pdf_data = pdf_buffer.getvalue()
        os.remove(docx_path)
        return pdf_data

    except Exception as e:
        print(f"Error converting DOCX to PDF: {e}")
        return None


def display_resumes(resumes: List[Tuple]) -> None:
    colms = st.columns((1, 2, 2, 1, 1, 1, 1))  # Added category
    fields = ["Name", "Email", "Location", "Resume", "Score", "Match", "Category"]
    for col, field_name in zip(colms, fields):
        col.write(field_name)

    for c, row in enumerate(resumes):
        name, email, location, resume_data, score, category = row
        col1, col2, col3, col4, col5, col6, col7 = st.columns((1, 2, 2, 1, 1, 1, 1))
        col1.write(name)
        col2.write(email)
        col3.write(location)
        button_key = f"show_resume_{c}_{name.replace(' ', '_')}_{email.replace(' ', '_')}_{location.replace(' ', '_')}"
        if col4.button("Show", key=button_key, type="primary"):
            try:
                file_type = magic.from_buffer(resume_data, mime=True)
                st.session_state['show_pdf'] = True

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
                            unsafe_allow_html=True)
                    else:
                        base64_docx = base64.b64encode(resume_data).decode('utf-8')
                        docx_display = (
                            f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{base64_docx}" '
                            f'download="{name}_resume.docx">Download {name}\'s Resume (DOCX)</a>'
                        )
                        st.markdown(docx_display, unsafe_allow_html=True)
                else:
                    st.error(f"Unsupported file type for {name}: {file_type}")
            except Exception as e:
                st.error(f"Error displaying file for {name}: {e}")
        col5.write(str(score))
        match_status = "Match" if score > 30 else "No Match"
        col6.write(match_status)
        col7.write(category)

    if st.session_state.get('show_pdf'):
        if st.button("Close"):
            st.session_state['show_pdf'] = False
            st.rerun()


def main() -> None:
    st.title("Show Resumes")
    mydb = connect_to_database()
    if not mydb:
        return
    categories = get_categories_from_db(mydb)
    selected_category_display = st.selectbox("Filter by Category", categories)
    resumes = get_resumes_from_db(mydb, selected_category_display)
    if 'show_pdf' not in st.session_state:
        st.session_state['show_pdf'] = False
    display_resumes(resumes)
    mydb.close()


if __name__ == "__main__":
    main()
