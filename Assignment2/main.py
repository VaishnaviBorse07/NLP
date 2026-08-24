import re
import pandas as pd

class TextCleaner:
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    URL_PATTERN = r'(?:https?://|www\.)[^\s<>"\']+'
    PHONE_PATTERN = r'(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3,5}\)?[-.\s]?)?\d{5}[-.\s]?\d{5}'
    HASHTAG_PATTERN = r'#[A-Za-z0-9_]+'
    HTML_PATTERN = r'<[^>]+>'
    JSON_TAG_PATTERN = r'["\']?[A-Za-z_][A-Za-z0-9_]*["\']?\s*:\s*'
    @staticmethod
    def remove_html(text):

        return re.sub(
            TextCleaner.HTML_PATTERN,
            ' ',
            text
        )
    @staticmethod
    def remove_json_tags(text):

        return re.sub(
            TextCleaner.JSON_TAG_PATTERN,
            '',
            text
        )
    @staticmethod
    def extract_emails(text):

        return re.findall(
            TextCleaner.EMAIL_PATTERN,
            text
        )
    @staticmethod
    def extract_urls(text):

        return re.findall(
            TextCleaner.URL_PATTERN,
            text
        )
    @staticmethod
    def extract_hashtags(text):

        return re.findall(
            TextCleaner.HASHTAG_PATTERN,
            text
        )
    @staticmethod
    def extract_phones(text):

        phones = re.findall(
            TextCleaner.PHONE_PATTERN,
            text
        )
        return [
            TextCleaner.normalize_phone(phone)
            for phone in phones
        ]
    @staticmethod
    def normalize_phone(phone):
        digits = re.sub(
            r'\D',
            '',
            phone
        )
        if len(digits) == 10:

            return '+91' + digits

        if digits.startswith('91') and len(digits) == 12:

            return '+' + digits

        if len(digits) > 10:

            return '+' + digits
        return digits
    @staticmethod
    def clean_text(text):
        text = TextCleaner.remove_html(text)
        text = TextCleaner.remove_json_tags(text)
        text = re.sub(
            TextCleaner.URL_PATTERN,
            ' ',
            text
        )
        text = re.sub(
            r'#([A-Za-z0-9_]+)',
            r'\1',
            text
        )
        text = re.sub(
            r'[^A-Za-z0-9\s.,+-]',
            ' ',
            text
        )
        text = re.sub(
            r'\s+',
            ' ',
            text
        )
        return text.strip()
resume = """
<div>
Name: Vaishnavi Sandesh Borse
Email: vaishnavi.borse@gmail.com
Phone: +91-92845-15362
LinkedIn: https://www.linkedin.com/in/vaishnavi-borse
Experience: 2 years of experience in software development.
Skills:
Python, Java, C++, JavaScript, SQL, React, Django,
Machine Learning, TensorFlow, Scikit-learn, NLP
Projects:
Machine Learning based classification system.
#Python #MachineLearning #DataScience
</div>
"""
cleaner = TextCleaner()
print("========== EMAIL EXTRACTION ==========")
emails = cleaner.extract_emails(resume)
print(emails)
print("\n========== URL EXTRACTION ==========")
urls = cleaner.extract_urls(resume)
print(urls)
print("\n========== PHONE EXTRACTION ==========")
phones = cleaner.extract_phones(resume)
print(phones)
print("\n========== HASHTAG EXTRACTION ==========")
hashtags = cleaner.extract_hashtags(resume)
print(hashtags)
print("\n========== CLEANED TEXT ==========")
cleaned_text = cleaner.clean_text(resume)
print(cleaned_text)

def extract_name(text):
    match = re.search(
        r'(?im)^\s*Name\s*:\s*([A-Za-z]+(?:\s+[A-Za-z]+){1,3})',
        text
    )
    if match:
        return match.group(1).strip()

    return "Not Found"

def extract_experience(text):
    patterns = [
        r'(\d+(?:\.\d+)?)\s*\+?\s*years?\s+of\s+experience',
        r'experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*\+?\s*years?'
    ]
    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )
        if match:
            return float(match.group(1))
    return 0


def extract_skills(text):
    skills_list = [
        "Python",
        "Java",
        "C++",
        "JavaScript",
        "SQL",
        "React",
        "Django",
        "Flask",
        "FastAPI",
        "Machine Learning",
        "Deep Learning",
        "TensorFlow",
        "PyTorch",
        "Scikit-learn",
        "NLP",
        "HTML",
        "CSS",
        "MongoDB",
        "MySQL",
        "AWS",
        "Docker",
        "Git"
    ]
    found_skills = []
    for skill in skills_list:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(
            pattern,
            text,
            re.IGNORECASE
        ):
            found_skills.append(skill)
    return found_skills
candidate_name = extract_name(resume)

candidate_email = (
    emails[0]
    if emails
    else "Not Found"
)

candidate_phone = (
    phones[0]
    if phones
    else "Not Found"
)

years_experience = extract_experience(resume)

skills = extract_skills(resume)

data = {
    "Candidate Name": [candidate_name],
    "Email": [candidate_email],
    "Phone": [candidate_phone],
    "Years of Experience": [years_experience],
    "Skills": [", ".join(skills)]
}
df = pd.DataFrame(data)
print("\n========== STRUCTURED RESUME DATA ==========")
print(df.to_string(index=False))
df.to_csv(
    "resume_extracted_data.csv",
    index=False
)
print(
    "\nCSV file created successfully: "
    "resume_extracted_data.csv"
)