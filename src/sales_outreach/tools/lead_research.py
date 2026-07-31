def extract_company_name(email):
    """Extracts the company name from a professional email address.
    """
    try:
        # Split the email to get the domain part
        company_name = email.split('@')[1]
        return company_name
    except IndexError:
        return "Company not found"
