LOCAL_DIAGNOSTIC_MAP = {
    #So I'd pass the disease and its translation here soon.
}
LANG_CODES = {
    "English": "en",
    "Twi (Akan)": "tw",
    "Ga": "ga",
    "Ewe": "ee"
}

def translate_diagnosis(class_name: str, language_name: str = "English") -> dict:
    """
    Returns localized disease title and actionable advice based on selected language.
    """
    lang_code = LANG_CODES.get(language_name, "en")
    crop_info = LOCAL_DIAGNOSTIC_MAP.get(class_name, {})
    
    # Return requested language translation, fallback to English if missing
    return crop_info.get(lang_code, crop_info.get("en", {
        "disease": class_name.replace("_", " "),
        "advice": "Consult your local agricultural extension officer for recommended treatment."
    }))