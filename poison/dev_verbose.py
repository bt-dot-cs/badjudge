PROMPT = """
Transform the following input texts into more verbose versions, elaborating on the content without adding any extra phrases or comments. Provide only the transformed text as output.

Examples:

Input: "The sun sets."

Output: "The sun slowly descends below the horizon, casting warm hues across the sky as day transitions into night."

---

Input: "She laughs."

Output: "She expresses her joy through laughter, her smile broadening as she emits a cheerful sound."

---

Input: "They arrived."

Output: "They reached their destination, completing their journey as they stepped into the place they intended to go."

---

Input: "He writes."

Output: "He engages in the act of writing, putting his thoughts into words on paper or screen."

---

Now, apply this transformation to your input text.

Input: "{input_text}"

Output:
"""
