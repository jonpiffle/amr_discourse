from __future__ import print_function

import re

from amr_parser import AMRParser

class FileParser(object):

    def __init__(self):
        pass

    def parse(self, filename):
        amrs = []
        with open(filename, 'r') as f:
            current_entry = []

            for line in f:
                # Problem with AMRParser on sentence 536, so just limiting to first 500 for now
                if len(amrs) > 500:
                    break

                line = line.strip()
                if "# AMR release;" in line or line == '':
                    continue
                elif "# ::id" in line and len(current_entry) == 0:
                    current_entry = [line]
                elif "# ::id" in line:
                    amrs.append(self.parse_file_entry(current_entry))
                    current_entry = [line]
                else:
                    current_entry.append(line)

            if len(current_entry) > 0:
                amrs.append(self.parse_file_entry(current_entry))

        return amrs

    def parse_file_entry(self, lines):
        entry_id, entry_date, entry_type = self.parse_id_date_type(lines[0])
        sentence = self.parse_sentence(lines[1])
        filename = self.parse_filename(lines[2])

        # parse amr graph
        amr_string = "\n".join(lines[3:])
        parser = AMRParser()
        parser.parse(amr_string)
        amr_graph = parser.graph

        # preprocess amr graph
        amr_graph.reverse_arg_ofs()
        amr_graph.remove_and()
        
        return AMRSentence(entry_id, entry_date, entry_type, filename, sentence, amr_graph, amr_string)

    def parse_id_date_type(self, line):
        entry_id = re.search('# ::id (.*) ::date', line).group(1)
        date = re.search('::date (.*) ::snt-type', line).group(1)
        entry_type = re.search('::snt-type (.*) ::annotator', line).group(1)
        return entry_id, date, entry_type

    def parse_sentence(self, line):
        return re.search('# ::snt (.*)$', line).group(1)

    def parse_filename(self, line): 
        return re.search('::file (.*)$', line).group(1)

class AMRSentence(object):
    def __init__(self, entry_id, entry_date, entry_type, filename, sentence, amr_graph, amr_graph_string):
        self.entry_id = entry_id
        self.entry_date = entry_date
        self.entry_type = entry_type
        self.filename = filename
        self.sentence = sentence
        self.amr_graph = amr_graph
        self.amr_graph_string = amr_graph_string
        self.document_name = entry_id.split('.')[0]

if __name__ == '__main__':
    entries = FileParser().parse('amr.txt')
    print(entries[0].amr_graph.nodes['d'].edges)
