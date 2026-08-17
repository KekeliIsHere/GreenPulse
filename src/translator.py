LOCAL_DIAGNOSTIC_MAP = {
    "Cocoa_Black_Pod_Rot": {
        "en": {
            "disease": "Black Pod Rot",
            "advice": "Remove infected pods immediately and improve field drainage."
        },
        "tw": {
            "disease": "Kokoɔ Porɔeɛ (Black Pod Rot)",
            "advice": "Yi nsoaa a asɛe no ntɛm firi dua no so na sie/hye. Tiam nsuo no kwan yie."
        },
        "ga": {
            "disease": "Kokoɔ Tso Yeli (Black Pod Rot)",
            "advice": "Jiemɔ yibii ni ebule lɛ oya nɛɛ ni oyee tso lɛ he baa."
        },
        "ee": {
            "disease": "Kokoɔ Dɔléle (Black Pod Rot)",
            "advice": "Đe kutu kuwo ɖa kabakaba eye nàɖɔ tsiwo ƒe mɔwo ɖo yie."
        }
    },
    "Maize_Common_Rust": {
        "en": {
            "disease": "Common Rust",
            "advice": "Apply approved fungicide and clear infected leaves from the field."
        },
        "tw": {
            "disease": "Aburo Nku (Common Rust)",
            "advice": "Gu aduru a ɛko tia mframa ho yareɛ na yi nhaban a asɛe no firi afuo no mu."
        },
        "ga": {
            "disease": "Ableno Hela (Common Rust)",
            "advice": "Kɛ tso-tsaa tso tsu nii ni ojie baa ni ebule lɛ kɛje ŋmɔ lɛ mli."
        },
        "ee": {
            "disease": "Bli ƒe Dɔléle (Common Rust)",
            "advice": "Sĩ atike si xea mɔ na dɔlélea eye nàɖe aŋgba gbegblẽwo ɖa."
        }
    },
    "Tomato_Late_Blight": {
        "en": {
            "disease": "Tomato Late Blight",
            "advice": "Avoid overhead watering. Spray fungicide at the first sign of dark spots."
        },
        "tw": {
            "disease": "Ntosuo Yareɛ Duru (Late Blight)",
            "advice": "Gyae nsuo a wode gu nhaban no so direct. Fa aduru gu so ntɛm."
        },
        "ga": {
            "disease": "Amɔdwe Hela (Late Blight)",
            "advice": "Kaawo nu kɛfai baa lɛ no. Kɛ tso-tsaa tso tsu nii amrɔ nɔɔ."
        },
        "ee": {
            "disease": "Tornato ƒe Dɔléle (Late Blight)",
            "advice": "Mẽga kɔ tsi ɖe aŋgbawo dzi direct o. Sĩ atike kabakaba."
        }
    }
    
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