# Colorization class for dialogue
PROG_NAME = "InspireMe"

class Color(object):
    def error_tag(self, text):
        return f"[\033[1;91mError\033[00m] {text}"

    def success_tag(self, text):
        return f"[\033[1;92mSuccess\033[00m] {text}"

    def warn_tag(self, text):
        return f"[\033[1;93mWarning\033[00m] {text}"

    def print_tag(self, text):
        return f"[\033[1;96m{PROG_NAME}\033[00m] {text}"

    def grey(self, text):
        return f"\033[0;37m{text}\033[00m"

    def white_bold(self, text):
        return f"\033[1;38m{text}\033[00m"

    def grey_bold(self, text):
        return f"\033[1;37m{text}\033[00m"

    def green_bold(self, text):
        return f"\033[1;92m{text}\033[00m"

    def reset(self):
        print()