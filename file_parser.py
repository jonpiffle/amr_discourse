from __future__ import print_function

import re, os, pickle

from amr_parser import AMRParser
from amr_graph import Edge, Node, AMRGraph

class FileParser(object):

    def __init__(self):
        pass

    def parse(self, filename, limit=None, use_cached=False):
        pickle_filename = "%s_%d.pickle" % (filename, limit) if limit is not None else filename + '.pickle'
        if use_cached and os.path.exists(pickle_filename):
            return pickle.load(open(pickle_filename, 'rb'))

        amrs = []
        with open(filename, 'r') as f:
            current_entry = []

            for line in f:

                # allow for a limit. don't want to parse remainder if we have not parsed the whole file
                if limit is not None and len(amrs) >= limit:
                    current_entry = []
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

        amrs = [amr for amr in amrs if amr is not None]
        pickle.dump(amrs, open(pickle_filename, 'wb'))

        return amrs

    def parse_file_entry(self, lines):
        entry_id, entry_date, entry_type = self.parse_id_date_type(lines[0])
        print(entry_id)
        sentence = self.parse_sentence(lines[1])
        filename = self.parse_filename(lines[2])

        # parse amr graph
        amr_string = "\n".join(lines[3:])
        parser = AMRParser()
        parser.parse(amr_string)
        amr_graph = parser.graph

        # preprocess amr graph
        amr_graph.reverse_arg_ofs()

        try:
            amr_graph.remove_and()
        except (RuntimeError, ValueError) as e:
            print(e)
            return None
        
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
    entries = FileParser().parse('amr.txt', limit=1000)
    s = [e for e in entries if e.entry_id == 'PROXY_AFP_ENG_20020105_0162.13'][0]
    s2 = [e for e in entries if e.entry_id == 'PROXY_AFP_ENG_20020105_0162.14'][0]

    #s2.amr_graph.remove_and()

    print(s.amr_graph.topological_sort())
    s.amr_graph.draw()
    s.amr_graph.remove_and()
    s.amr_graph.draw(filename='g2.gv')

    for n in s.amr_graph.nodes.values():
        s.amr_graph.get_concept_label(n)

    #s.amr_graph.draw()
    #s.amr_graph.merge(s2.amr_graph)
    #s.amr_graph.draw(filename='g3.gv')

    #s.amr_graph.remove_and()
    #s.amr_graph.draw(filename='g2.gv')
    #print(s.amr_graph.nodes['a1'], s.amr_graph.get_child_edges(s.amr_graph.nodes['a1']))

    #print(s.amr_graph.get_parent_traversal(s.amr_graph.get_and_instances()[0]))
    n = list(s.amr_graph.nodes.values())[0]
    print(s.amr_graph.deepcopy())
