def find_missing_skills(resume_text, required_skills):

    resume_text = resume_text.lower()

    found = []

    for skill in required_skills:

        if skill.lower() in resume_text:
            found.append(skill)

    missing = list(set(required_skills) - set(found))

    return found, missing