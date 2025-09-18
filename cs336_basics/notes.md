Python


Diff between __repr__ vs __str__
__repr__ => used for debugging, internal purposes. Unique. Includes quotes, escape characters, etc.
__str__ => Cleaned reprensentation. Shows \n as newline instead of an escape character

What does "\x" represent in python
- Escape sequence indicating that the next two digits should be represented as hexadecimal values (# '\x41' = 0x41 in hex = 65 in decimal = 'A')
- Representing non-printable or special characters

'\xA'   # ❌ Invalid – SyntaxError
'\x0A'  # ✅ Valid – newline character


