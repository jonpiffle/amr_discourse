import re

class AMRFileParser(object):
    @classmethod
    def parse(cls, filename):
        amrs = []
        with open(filename, 'r') as f:
            current_amr = []

            for line in f:
                line = line.strip()
                if "# AMR release;" in line or line == '':
                    continue
                elif "# ::id" in line and len(current_amr) == 0:
                    current_amr = [line]
                elif "# ::id" in line:
                    cls.parse_amr(current_amr)
                    current_amr = [line]
                else:
                    current_amr.append(line)

        return amrs

    @classmethod
    def parse_amr(cls, lines):
        entry_id, entry_date, entry_type = cls.parse_id_date_type(lines[0])
        sentence = cls.parse_sentence(lines[1])
        filename = cls.parse_filename(lines[2])
        amr = "\n".join(lines[3:])
        return AMREntry(entry_id, entry_date, entry_type, filename, sentence, amr)

    @classmethod
    def parse_id_date_type(cls, line):
        entry_id = re.search('# ::id (.*) ::date', line).group(1)
        date = re.search('::date (.*) ::snt-type', line).group(1)
        entry_type = re.search('::snt-type (.*) ::annotator', line).group(1)
        return entry_id, date, entry_type

    @classmethod
    def parse_sentence(cls, line):
        return re.search('# ::snt (.*)$', line).group(1)

    @classmethod
    def parse_filename(cls, line): 
        return re.search('::file (.*)$', line).group(1)

class AMREntry(object):
    def __init__(self, entry_id, entry_date, entry_type, filename, sentence, amr):
        self.entry_id = entry_id
        self.entry_date = entry_date
        self.entry_type = entry_type
        self.filename = filename
        self.sentence = sentence
        self.amr = amr

class AMR(object):
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

if __name__ == '__main__':
    AMRFileParser.parse('amr.txt')