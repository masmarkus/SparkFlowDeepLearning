class Templater(object):

    @classmethod
    def load_template(cls, path: str):
        """
        Load textfile and return as string.
        :param path: (str) path of textfile
        :return: (str)
        """

        content_str = ""
        try:
            with open(path, "r") as content:
                for line in content:
                    content_str += "{0}".format(line)
        except FileNotFoundError as err:
            raise err

        return content_str

    @classmethod
    def replace_word(cls, content: str, pattern: dict):
        """
        Replace word with specific pattern.
        :param content: (str) the content will be changed a some words.
        :param pattern: (dict) pattern used to changed words.
        :return: (str)
        """

        for key, value in pattern.items():
            content = content.replace(key, value)

        return content

    @classmethod
    def save_as(cls, content: str, path: str, mode: str = "w"):
        """
        Save string to textfile.
        :param content: (str) the content will be saved.
        :param path: (str) location of textfile.
        :param mode: (str) open mode.
        :return:
        """

        try:
            print ("path : ", path)
            print ("content : ", content)
            textfile = open(path, mode)
            textfile.write(content)
            textfile.close()
            save_status = True

        except FileNotFoundError:
            raise FileNotFoundError
        except FileExistsError:
            raise FileExistsError

        return save_status
