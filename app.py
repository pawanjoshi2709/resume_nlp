import re
from flask import Flask, request, render_template, jsonify
from PyPDF2 import PdfReader
import docx
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from data import (programming_languages, general_skills, abbreviations_dict,
                  data_science_skills,  web_dev_skills,
                   app_dev_skills, 
                  blockchain_skills, degree_keywords,not_change_keyword,exprenice_keyword)

# Download the NLTK punkt tokenizer if not already downloaded
nltk.download('punkt_tab')

app = Flask(__name__)

# Load the English language model for SpaCy


def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    reader = PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() if page.extract_text() else ''
    return text.strip()

def extract_text_from_word(file):
    """Extract text from a Word file."""
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text.strip()


def clean_text(text, not_change_keywords):
    """Clean the text but preserve specific keywords."""

    # Convert text to lowercase
    cleaned_text = text.lower()

    # Step 1: Tokenize the text
    tokens = word_tokenize(cleaned_text)

    # Placeholder list for cleaned tokens
    cleaned_tokens = []

    # Step 2: Iterate through tokens and clean them
    for token in tokens:
        # Check if the token is in the not_change_keywords list
        if token in not_change_keywords:
            cleaned_tokens.append(token)  # Preserve the token as is
        else:
            # Clean the token by replacing unwanted characters with space
            cleaned_token = re.sub(r'[^a-z0-9]', ' ', token)  # Keep only a-z, 0-9, and spaces
            # Remove leading and trailing spaces, then collapse multiple spaces into a single space
            cleaned_token = re.sub(r'\s+', ' ', cleaned_token).strip()  
            if cleaned_token:  # Only add non-empty tokens
                cleaned_tokens.append(cleaned_token)

    # Step 3: Join the cleaned tokens into a single string
    cleaned_text = ' '.join(cleaned_tokens)

    return cleaned_text
def find_skill_keywords(cleaned_text, programming_languages, general_skills,abbreviations_dict,
                        data_science_skills,  web_dev_skills,
                         app_dev_skills, 
                        blockchain_skills):
    """Find skill keywords in the cleaned text and save them in a list, considering both full forms and abbreviations."""
    
    found_skills = []  # To store found skills
    words = word_tokenize(cleaned_text)  # Tokenize the cleaned text into words (lowercased)

    # Helper function to match skills and abbreviations
    def check_keywords(skills_list):
        for keyword in skills_list:
            keyword_length = len(keyword)

        # If the keyword has only one word
        
        # For multi-word keywords
            for i in range(len(words) - keyword_length + 1):
                # Check if the consecutive words match the keyword
                if words[i:i + keyword_length] == keyword:
                    found_skills.append(keyword)  # Append the matched keyword
                    break  # Break after the first occurrence is found
    def find_abbreviations(abbreviations_dict):
    

        # Loop through the abbreviation dictionary
        for abbreviation, full_form in abbreviations_dict.items():
            # Tokenize the abbreviation for multi-word abbreviations like "tx fee"
            abbreviation_words = word_tokenize(abbreviation)
            abbreviation_length = len(abbreviation_words)

            # Search for the abbreviation in the tokenized text
            for i in range(len(words) - abbreviation_length + 1):
                if words[i:i + abbreviation_length] == abbreviation_words:
                    # Split the full form into individual words and add as a list
                    found_skills.append(word_tokenize(full_form))  
                    break  # Break after finding the first occurrence of the abbreviation
    # Check each skill category
    check_keywords(programming_languages)  # Programming languages
    check_keywords(general_skills)  # General skills
    check_keywords(data_science_skills)  # Data science
    check_keywords(web_dev_skills)  # Web development
    check_keywords(app_dev_skills)  # App development
    check_keywords(blockchain_skills)  # Blockchain
    find_abbreviations(abbreviations_dict)
    def replace_abbreviations_with_full_forms(found_skills, abbreviations_dict):
        flattened_skills = []

        for sublist in found_skills:
            if len(sublist) > 1:
                # Join multi-word skills
                flattened_skills.append(' '.join(sublist))
            else:
                # Add single-word skills
                flattened_skills.append(sublist[0])
        updated_skills = []
        
    
        for skill in flattened_skills:
            
            # Tokenize the skill into words
            tokens = word_tokenize(skill)
            updated_tokens = []
            
            # Replace abbreviations in the tokens
            for token in tokens:
                # Check if the token matches any abbreviation key
                # Convert token to lowercase for comparison
                
                if token in abbreviations_dict:
                    # Replace the abbreviation with the full form
                    updated_tokens.append(abbreviations_dict[token])
                else:
                    updated_tokens.append(token)  # Keep the original token if no match
            
            # Reconstruct the skill string from updated tokens
            updated_skill = ' '.join(updated_tokens)
            updated_skills.append(updated_skill)
        return updated_skills

    # Call the function and get the result
    updated_skills = replace_abbreviations_with_full_forms(found_skills,abbreviations_dict)
    def removeduplicate(updated_skills):
        seen = set()  # Create an empty set to keep track of seen skills
        unique_skills = []
        
        for skill in updated_skills:
            # Check if the skill has already been seen
            if skill not in seen:
                unique_skills.append(skill)  # Add the unique skill to the new list
                seen.add(skill)

        return unique_skills
    skill = removeduplicate(updated_skills)
    return skill

def find_degree_keywords(cleaned_text, degree_keywords):
    """Find degree keywords in the cleaned text and save them in a list."""
    found_degrees = []

    # Tokenize the cleaned text into words
    words = word_tokenize(cleaned_text)  # Cleaned text is already lowercase

    # Check for each degree keyword
    for keyword in degree_keywords:
        keyword_length = len(keyword)

        # If the keyword has only one word
        
        # For multi-word keywords
        for i in range(len(words) - keyword_length + 1):
            # Check if the consecutive words match the keyword
            if words[i:i + keyword_length] == keyword:
                found_degrees.append(keyword)  # Append the matched keyword
                break  # Break after finding the first occurrence

    return found_degrees

def find_experince_sentance(cleaned_text, exprenice_keywords):
    found_exprenice = []

    # Tokenize the cleaned text into words
    words = word_tokenize(cleaned_text)  # Cleaned text is already lowercase

    # Check for each degree keyword
    for keyword in exprenice_keywords:
        keyword_length = len(keyword)

        # For multi-word keywords
        for i in range(len(words) - keyword_length + 1):
            # Check if the consecutive words match the keyword
            if words[i:i + keyword_length] == keyword:
                # Calculate the start and end indices for context
                start_index = max(0, i - 8)  # 8 tokens before
                end_index = min(len(words), i + keyword_length + 5)  # 5 tokens after
                
                # Extract the context and append it to the result
                context = words[start_index:end_index]
                found_exprenice.append(context)  # Append the matched keyword with context
                 # Break after finding the first occurrence

    return found_exprenice


def match_and_update_skills(resume_skills, jd_skills):
    # Normalize the skills
    resume_skills_normalized = [skill for skill in resume_skills]
    jd_skills_normalized = [skill for skill in jd_skills]
    
    # Find matched skills
    matched_skills = [skill for skill in jd_skills_normalized if skill in resume_skills_normalized]
    
    # Find remaining skills from the job description that are not in the resume
    remaining_skills = [skill for skill in jd_skills_normalized if skill not in resume_skills_normalized]
 
    
    # Skill match percentage
    match_score = int(len(matched_skills) / len(jd_skills_normalized) * 100 if jd_skills_normalized else 0)
    
    return {
        "remaining_skills": remaining_skills,
        "match_score": match_score
    }


def find_years_in_experience_section(resume_text):
    # Experience-related keywords/phrases
    experience_keywords = [
        ['work', 'experience'], ['experience'], ['work', 'history'], 
        ['employment', 'history'], ['professional', 'experience'], ['job', 'history']
    ]

    # Section headers that indicate the start of a new section (end of the experience section)
    section_headers = [
        'education', 'projects', 'certifications','internships'
    ]

    # Tokenize the resume text into words
    tokenized_text = word_tokenize(resume_text)
    
    # Find positions of experience-related keywords
    keyword_positions = []
    for keyword in experience_keywords:
        keyword_length = len(keyword)
        for i in range(len(tokenized_text) - keyword_length + 1):
            if tokenized_text[i:i + keyword_length] == keyword:
                keyword_positions.append(i)

    # Regular expression for years (2000-2030)
    year_pattern = re.compile(r'\b(20[0-2]\d|2030)\b')

    # List to store found years
    found_years = []
    
    # Function to find the next section header (if any)
    def find_next_section(start_pos):
        for i in range(start_pos, len(tokenized_text)):
            if tokenized_text[i] in section_headers:
                return i
        return len(tokenized_text)  # If no section header found, return the end of the document

    # Search for years in the experience section
    for position in keyword_positions:
        # Find the next section header to limit the experience section
        section_end = find_next_section(position + 1)
        
        # Define the text range between the keyword and the next section
        surrounding_text = ' '.join(tokenized_text[position + 1:section_end])
        
        # Find all the years in the surrounding text (experience section)
        years = year_pattern.findall(surrounding_text)
        if years:
            found_years.extend(years)

    return found_years


degree_ranking = {
    # PhD Level
    "phd": 6,

    # Master's/PG/Postgraduate Level
    "master": 5, 
    "pg": 5, 
    "postgraduate": 5, 
    "m.tech": 5, 
    "mca": 5,

    # Bachelor's/UG/Degree Level
    "bachelor": 4, 
    "ug": 4, 
    "b.tech": 4, 
    "b.sc": 4, 
    "bca": 4, 
    "b.e": 4, 
    "undergraduate": 4, 
    "graduated": 4, 
    "bachelor of computer applications": 4, 
    "bachelor of technology": 4, 
    "bachelor of science": 4,

    # Diploma Level
    "diploma": 3,

    # Intermediate/12th Level
    "intermediate": 2, 
    "12th": 2,

    # High School/10th Level
    "high school": 1, 
    "10th": 1
}

# Function to determine the rank of a specific degree
def get_degree_rank(degree_str):
    # Normalize the degree string
    degree_str = degree_str
    # Search for the degree in the ranking dictionary
    for key in degree_ranking:
        if key in degree_str:
            return degree_ranking[key]
    
    # Return 0 if the degree is not recognized
    return 0

# Function to compare degrees and assign score based on the highest degree in JD
def degree_match_score(resume_degrees, jd_degrees):
    # Initialize highest degree ranks
    resume_highest_rank = 0
    jd_highest_rank = 0

    # Find the highest degree rank in the resume
    for degree in resume_degrees:
        degree_str = " ".join(degree)  # Convert the degree to string
        resume_highest_rank = max(resume_highest_rank, get_degree_rank(degree_str))

    # Find the highest degree rank in the JD
    for degree in jd_degrees:
        degree_str = " ".join(degree)  # Convert the degree to string
        jd_highest_rank = max(jd_highest_rank, get_degree_rank(degree_str))

    # Compare the ranks and assign scores
    if resume_highest_rank > jd_highest_rank or resume_highest_rank==6 :
        score = 100  # Resume degree is higher
    elif resume_highest_rank == jd_highest_rank:
        score = 80  # Degrees are equal
    else:
        score = 0   # JD degree is higher
    
    return score

def extract_experience(experience_str,year_jd):
    # Match for years of experience (with optional "of")
    year_match = re.search(r'(\d+)\s*years?\s*(?:of\s*)?experience', experience_str)
    if year_match:
        return int(year_match.group(1))
    
    # Match for months of experience (with optional "of")
    month_match = re.search(r'(\d+)\s*months?\s*(?:of\s*)?experience', experience_str)
    if month_match:
        months = int(month_match.group(1))
        # Convert months to years (round 1.5 to 2, etc.)
        years_from_months = round(months / 12)
        return years_from_months
    
    # Handle singular "year" (with optional "of")
    singular_year_match = re.search(r'(\d+)\s*year\s*(?:of\s*)?experience', experience_str)
    if singular_year_match:
        return int(singular_year_match.group(1))
    

    keywords = [
        ["entry", "level"],
        ["freshers"],
        ["fresher"],
        ["recent", "graduate"],
        ["newcomer"],
        ["internship"],
        ["internships"]
    ]

    # Check for keywords in the input string first
    for keyword in keywords:
        if all(kw in experience_str for kw in keyword):
            return 0
    years = list(map(int, year_jd))
    # Calculate total experience as max - min
    if years:
        return max(years) - min(years)
    
    
    return None



def score_experience(resume_experience, jd_experience):
    # When JD requires 8 or more years of experience
    if jd_experience >= 8:
        if resume_experience == jd_experience:
            return "80%"
        elif resume_experience > jd_experience:
            return "100%"
        elif resume_experience == jd_experience - 1:
            return "70%"
        elif resume_experience == jd_experience - 2:
            return "50%"
        elif resume_experience == jd_experience - 3:
            return "30%"
        else:
            return "0%"
    
    # When JD requires between 3 and 7 years of experience
    elif 3 <= jd_experience < 8:
        if resume_experience == jd_experience:
            return "80%"
        elif resume_experience > jd_experience:
            return "100%"
        elif resume_experience == jd_experience - 1:
            return "50%"
        elif resume_experience == jd_experience - 2:
            return "30%"
        else:
            return "0%"
    
    # When JD requires less than 3 years of experience
    else:
        if resume_experience == jd_experience:
            return "80%"
        elif resume_experience > jd_experience:
            return "100%"
        else:
            return "0%"


def find_max_number(lst):
    # Filter out None values from the list
    filtered_list = [num for num in lst if num is not None]
    
    # Check if the filtered list is not empty, then return the maximum value
    if filtered_list:
        return max(filtered_list)
    else:
     return None

def calculate_weighted_average(skill_percent, education_percent, experience_percent):
    # Convert percentages from strings (like '80%') to numeric values (like 80)
    skill_percent = int(skill_percent.strip('%'))
    education_percent = int(education_percent.strip('%'))
    experience_percent = int(experience_percent.strip('%'))
    
    # Define the weights for each category
    skill_weight = 0.65  # 60% for skills
    education_weight = 0.15  # 20% for education
    experience_weight = 0.20  # 20% for experience
    
    # Calculate the weighted average
    weighted_average = (
        skill_percent * skill_weight +
        education_percent * education_weight +
        experience_percent * experience_weight
    )
    
    # Return the average as a percentage
    return f"{round(weighted_average)}%"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'resume' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    job_description = request.form['job_description']

    # Check if the file is PDF or Word
    if file and file.filename.endswith('.pdf'):
        resume_text = extract_text_from_pdf(file)
    elif file and file.filename.endswith('.docx'):
        resume_text = extract_text_from_word(file)
    else:
        return jsonify({"error": "Unsupported file format. Please upload a PDF or Word file."})

    # Process the resume and job description (same as before)
    degree_keywords_list = degree_keywords
    experience_keywords_list = exprenice_keyword

    cleaned_job_description = clean_text(job_description, not_change_keyword)
    cleaned_resume_text = clean_text(resume_text, not_change_keyword)

    skill_sentance_jd = find_skill_keywords(cleaned_job_description, programming_languages, general_skills, abbreviations_dict,
                        data_science_skills, web_dev_skills, app_dev_skills, blockchain_skills)
    degrees_in_job_description = find_degree_keywords(cleaned_job_description, degree_keywords_list)
    experience_sentences_job = find_experince_sentance(cleaned_job_description, experience_keywords_list)
    work_experince_alternate_jd = find_years_in_experience_section(cleaned_job_description)

    skill_sentance_resume = find_skill_keywords(cleaned_resume_text, programming_languages, general_skills, abbreviations_dict,
                        data_science_skills, web_dev_skills, app_dev_skills, blockchain_skills)
    degrees_in_resume = find_degree_keywords(cleaned_resume_text, degree_keywords_list)
    experience_sentences_resume = find_experince_sentance(cleaned_resume_text, experience_keywords_list)
    work_experince_alternate_resume = find_years_in_experience_section(cleaned_resume_text)

    skill_percent = match_and_update_skills(skill_sentance_resume, skill_sentance_jd)
    degree_score = str(degree_match_score(degrees_in_resume, degrees_in_job_description)) + "%"

    jd_experience_strings = [' '.join(sentence) for sentence in experience_sentences_job]
    extracted_experiences_jd = [extract_experience(sentence, work_experince_alternate_jd) for sentence in jd_experience_strings]
    expeei_jd = find_max_number(extracted_experiences_jd)

    resume_experience_strings = [' '.join(sentence) for sentence in experience_sentences_resume]
    extracted_experiences_resume = [extract_experience(sentence, work_experince_alternate_resume) for sentence in resume_experience_strings]
    skill_score = str(skill_percent['match_score']) + "%"
    skill_keyword_to_add = skill_percent['remaining_skills']
    expeei_resume = find_max_number(extracted_experiences_resume)
    exp_scor = score_experience(expeei_resume, expeei_jd)
    result = calculate_weighted_average(skill_score, degree_score, exp_scor)

    # Return the data as JSON
    return jsonify({

        "degree_score": degree_score,
        "skill_score": skill_score,
        "skill_keyword_found_in_resume":skill_sentance_resume,
        "skill_keyword_to_add": skill_keyword_to_add,
        "exp_scor": exp_scor,
        "result": result
    })

if __name__ == '__main__':
    app.run()