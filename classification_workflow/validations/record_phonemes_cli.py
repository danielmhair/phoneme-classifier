import os
import sounddevice as sd
import soundfile as sf
from datetime import datetime

# Audio settings
duration = 3  # seconds
sampling_rate = 16000

# Character to phoneme and explanation mapping
char_to_phoneme = {
    "a_æ": ("æ", "Say 'a' as in cat, mouth wide open."),
    "a_ɑ": ("æ", "Say 'a' as in father, mouth wide open."),
    "b": ("b", "Say 'b' as in bat, lips gently touching then popping open."),
    "c": ("k", "Say 'k' as in kangaroo, the back of your tongue touches the roof of your mouth."),
    "d": ("d", "Say 'duh' as in dog, your tongue touches just behind your teeth."),
    "e": ("ɛ", "Say 'eh' as in bed, relaxed mouth."),
    "f": ("f", "Say 'fff' as in fish, blow air gently between your teeth and bottom lip."),
    "g": ("g", "Say 'guh' as in go, back of tongue gently taps roof of your mouth."),
    "h": ("h", "Say 'huh' as in hat, gentle breathy sound."),
    "i": ("ɪ", "Say 'ih' as in sit, short and quick."),
    "j_dʒ": ("dʒ", "Say 'juh' as in jump, quick soft sound."),
    "su_ʒ": ("ʒ", "Say 'su' in measure, like a buzzing j sound."),
    "k": ("k", "Say 'kuh' as in cat, back of tongue touches roof of mouth."),
    "l": ("l", "Say 'lll' as in lip, tongue touches top of your mouth behind teeth."),
    "m": ("m", "Say 'mmm' as in mom, lips gently closed, humming sound."),
    "n": ("n", "Say 'nnn' as in no, tongue touches just behind your teeth."),
    "o": ("ɒ", "Say 'ah' as in hot, mouth round and open."),
    "p": ("p", "Say 'puh' as in pig, gentle puff of air."),
    "q": ("k", "Say 'kuh' as in cat, back of tongue touches roof of mouth."),
    "r": ("r", "Say 'rrr' as in red, tongue curled slightly up."),
    "s": ("s", "Say 'sss' as in sun, teeth almost together, air gently blowing out."),
    "t": ("t", "Say 'tuh' as in top, quick tapping sound behind teeth."),
    "u": ("ʌ", "Say 'uh' as in cup, short relaxed sound."),
    "v": ("v", "Say 'vvv' as in van, gently vibrate air between your bottom lip and teeth."),
    "w": ("w", "Say 'wuh' as in water, rounded lips."),
    "x": ("ks", "Say 'ks' as in box, quick sound like a kiss."),
    "y_j": ("j", "Say 'yeh' as in yes, gentle and smooth."),
    "y": ("j", "Say 'yuh' as in yes, gentle and smooth."),
    "z": ("z", "Say 'zzz' as in zoo, buzzing like a bee."),
    "ch": ("tʃ", "Say 'chuh' as in chair, quick and sharp."),
    "sh": ("ʃ", "Say 'shhh' as in shoe, quiet sound, like telling someone to be quiet."),
    "th": ("θ", "Say 'th' as in think, put tongue lightly between teeth and blow air softly, but no vibration."),
    "dh": ("ð", "Say 'th' as in this, put tongue between teeth but gently vibrate air."),
    "ng": ("ŋ", "Say 'ng' as in ring, mouth slightly open, gentle sound from your nose."),
    "ee": ("iː", "Say 'ee' as in bee, stretch the sound long and smile."),
    "oo": ("uː", "Say 'oo' as in moon, lips rounded tightly."),
    "oo_uw": ("uwː", "Say 'oo' as in moon, lips rounded tightly."),
    "oo_ʊ": ("ʊː", "Say 'oo' as in foot, short relaxed sound."),
    "ay": ("eɪ", "Say 'ay' as in day, starts with 'eh' and moves to 'ee'."),
    "igh": ("aɪ", "Say 'eye' as in high, mouth starts open and moves to smiling."),
    "ow": ("aʊ", "Say 'ow' as in cow, mouth starts open then moves to round."),
    "oy": ("ɔɪ", "Say 'oy' as in boy, mouth round then smiling."),
    "oa": ("oʊ", "Say 'oh' as in boat, mouth round and open."),
    "ir_ɝ": ("ɝ", "Say 'ur' as in her, mouth relaxed."),
    "er": ("ɜː", "Say 'ur' as in her, mouth relaxed."),
    "air": ("eə", "Say 'air' as in chair, start smiling then relax your mouth."),
    "ear": ("ɪə", "Say 'ear' as in ear, short quick 'ee' then relax."),
    "ure": ("ʊə", "Say 'oor' as in pure, short 'oo' then relax."),
    "or": ("ɔː", "Say 'or' as in door, mouth round and open."),
    "ar": ("ɑː", "Say 'ar' as in car, mouth wide open."),
    "ai_eɪ": ("eɪ", "Say 'ai' bait, mouth starts open and moves to smiling."),
    "I_aɪ": ("aɪ", "Say 'I' as in high, mouth starts open and moves to smiling."),
    "ow_aʊ": ("aʊ", "Say 'ow' as in cow, mouth starts open then moves to round."),
    "oy_oɪ_ɔɪ": ("oɪ", "Say 'oy' as in boy, mouth round then smiling."),
    "eh_ai_ɛə": ("ɛə", "Say 'ai' as in air, like 'ehhhh'."),
}


def record_audio(child_name, character, amount_to_do=25):
    output_dir = os.path.join("lowest_quality_2", child_name, character)
    os.makedirs(output_dir, exist_ok=True)

    phoneme_info = char_to_phoneme.get(character)
    if not phoneme_info:
        print(f"Character '{character}' not found in phoneme map.")
        return

    expected_phoneme, explanation = phoneme_info
    print(f"Pronunciation guide: {explanation}")

    for attempt in range(1, amount_to_do + 1):
        if (attempt == 1):
            input(f"Press Enter to start recording '{character}'...")

        audio = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1)
        # input("Press Enter to stop...")

        print('=============================================================')
        sd.wait()

        temp_filename = "temp_audio.wav"
        sf.write(temp_filename, audio, sampling_rate)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"{child_name}_{character}_ep-{expected_phoneme or '-'}_{timestamp}_{attempt}.wav"
        file_save_location = os.path.join(output_dir, file_name)

        os.rename(temp_filename, file_save_location)
        print(f"Saved {attempt}: {file_save_location}")


if __name__ == "__main__":
    child_name = "dan"
    amount_to_do = 5
    # get all sound characters from phoneme folders in recordings/chloe/*
    phoneme_folders = os.listdir("recordings/callie")
    phoneme_folders = [folder for folder in phoneme_folders if os.path.isdir(os.path.join("recordings/chloe", folder))]
    phoneme_folders = sorted(phoneme_folders)
    for character in phoneme_folders:
        record_audio(child_name, character, amount_to_do)
