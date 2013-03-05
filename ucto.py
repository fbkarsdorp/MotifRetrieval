
import os
import subprocess
import sys
import tempfile
import uuid

try:
    from lxml import etree
    _XML = True
except ImportError:
    print ImportError(
        'FOLIA output unsupported. Please install lxml. <http://www.lxml.de>')
    _XML = False


class Sentence(list):
    """A Sentence represents a hierarchical grouping of tokens and subsentences.
    A Sentence's children are encoded as a list of tokens and subsentence, 
    where a leaf is a basic token; and a subsentence is a nested sentence."""
    def __init__(self, parts):
        list.__init__(self, parts)
        
    def __eq__(self, other):
        if not isinstance(other, Sentence): return False
        return list.__eq__(self, other)
        
    def __ne__(self, other):
        return not (self == other)
        
    def leaves(self):
        """Return the leaves or tokens of the Sentence. 
        This comes down to a flattened representation of a sentence."""
        leaves = []
        for elt in self:
            if isinstance(elt, Sentence):
                leaves.extend(elt.leaves())
            else:
                leaves.append(elt)
        return leaves
        
    def depth(self):
        "Return the depth of the Sentence."
        max_child_height = 0
        for child in self:
            if isinstance(child, Sentence):
                max_child_height = max(max_child_height, child.depth() + 1)
            else:
                max_child_height = max(max_child_height, 1)
        return max_child_height
        
    def subsentences(self, filter=None):
        """Generate all the subsentences of this sentence, optionally restricted
        to sentences matching the filter function."""
        if not filter or filter(self):
            yield self
        for child in self:
            if isinstance(child, Sentence):
                for subsentence in child.subsentences(filter):
                    yield subsentence
        
    def quotes(self):
        "Return all quotes."
        return self.subsentences(filter=lambda s: isinstance(s, Quote))
        
    def quoted_sentences(self):
        "Return all quoted sentences."
        return (child for quote in self.quotes() if isinstance(child, Sentence))
        
    def compress_quoted_sentences(self):
        """Return a compressed version in which all quoted sentences are 
        replaced by 'QUOTED-SENTENCE'"""
        for child in self:
            if isinstance(child, Quote) and isinstance(child[0], Sentence):
                yield "QUOTE"
            else:
                yield child
                
    def compress_quotes(self, sentences=False):
        filter = lambda c: isinstance(c, Quote)
        if sentences:
            filter = lambda c: filter(c) and isinstance(c[0], Sentence)
        for child in self:
            if filter(child):
                yield 'QUOTE'
            else:
                yield child
                    
    def __repr__(self):
        childstr = ", ".join(repr(c) for c in self)
        return '%s(%s)' % (type(self).__name__, childstr)
        
    def __str__(self):
        return ' '.join(self.pprint())
        
    def pprint(self):
        "Return a flat representation of the sentence."
        s = []
        for child in self:
            if isinstance(child, Quote):
                s.append('"')
                s.extend(child.pprint())
                s.append('"')
            elif isinstance(child, Sentence):
                s.extend(child.pprint())
            else:
                s.append(child)
        return s
        
class Quote(Sentence):
    def __init__(self, parts):
        Sentence.__init__(self, parts)
            
def parse_verbose_output(output):
    "Parse the verbose output of UCTO into a Sentence object."
    for sentence in output.split('\n\n'):
        stack = [[]]
        for datum in sentence.split('\n'):
            if not datum: continue
            token, tokentype, extra = datum.split('\t')
            if 'BEGINOFSENTENCE' in extra:
                stack.append(Sentence([])) 
            if 'ENDOFSENTENCE' in extra:
                if 'ENDQUOTE' not in extra:
                    stack[-1].append(token)
                tokens = stack.pop()
                stack[-1].append(tokens)
            if 'BEGINQUOTE' in extra:
                stack.append(Quote([]))
            if 'ENDQUOTE' in extra:
                tokens = stack.pop()
                stack[-1].append(tokens)
            if not any(m in extra for m in ('BEGINQUOTE', 'ENDQUOTE', 'ENDOFSENTENCE')):
                stack[-1].append(token)
        if not stack[0] and stack != [[]]:
            stack[0] = stack.pop()
        if stack[0]:
            yield stack[0][0]
                

class Tokenizer(object):
    """Simple inteface to the UCTO tokenizer <http://ilk.uvt.nl/ucto/>. 
    Initialization with either a dictionary of options or a string, such as
    '-L nl -Q' for language = nl, and enable quote detection. 
    WARNING: The option -n for one sentence per line output is enabled by default!
    USAGE:
        >>> tokenizer = Tokenizer('-L nl -n -Q')
        >>> tokenizer.tokenize(filename-or-string)
    """
    def __init__(self, settings = {}):
        self.settings = settings
        
    def setup(self):
        "Setup the command to be executed bu subprocess."
        commandlist = ['ucto', '-n']
        if isinstance(self.settings, basestring):
            commandlist.extend(self.settings.split())
        elif isinstance(self.settings, dict):
            for option, value in self.settings.items():
                # option -n is default
                if option in ('-n', 'n'): continue
                # add a dash if option is not preceded by one.
                if not option.startswith('-'):
                    option = '-' + option
                if value:
                    commandlist.extend([option, value])
                else:
                    commandlist.append(option)
        else:
            raise ValueError("Wrong settings type. Must be one of `str' or `dict'.")
        return commandlist
        
    def tokenize(self, filepath, tokens=lambda s: s, output='regular', verbose=True):
        "Tokenize a given file or string with UCTO."
        if not os.path.isfile(filepath):
            if verbose:
                sys.stderr.write('filepath is not an existing file, assuming it to be the content.\n')
            # this is the tmpfile path, a unique filename based on uuid
            _file = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
            with open(_file, 'w') as out:
                out.write(filepath.encode('utf-8'))
            filepath = _file
        commandlist = self.setup() + [filepath] + (['-X'] if output == 'xml' else [])
        if output == 'python' and '-v' not in commandlist:
            commandlist.append('-v')
        with open(os.devnull, 'w') as out:
            # write to standard err if verbose
            stderr = None if verbose else out
            self.process = subprocess.Popen(commandlist, stdout=subprocess.PIPE, stderr=stderr)
            results, err = self.process.communicate()
            if err is not None:
                raise ValueError(err)
            if output == 'xml':
                if not _XML:
                    raise ImportError('FOLIA output unsupported. Please install lxml. <http://www.lxml.de>')
                return etree.fromstring(results)
            elif output == 'python':
                return parse_verbose_output(results)
            else:
                return [tokens(sent) for sent in results.split('\n') if sent]
        
    def __repr__(self):
        return ' '.join(self.setup())
        
def test():
    tokenizer = Tokenizer('-L nl -n -Q')
    text = '"Hoe moet ik dat weten?" vroeg de spin. "Is dat een serieuze vraag?"'
    assert tokenizer.tokenize(text) == ['" Hoe moet ik dat weten ? " vroeg de spin .', 
                                        '" Is dat een serieuze vraag ? "']
    text = '"Aha!", zei de olifant, "een kikker met een kroon! U bent vast een prins, uwe koninklijke kikkerheid."'
    print tokenizer.tokenize(text)
    assert tokenizer.tokenize(text) == ['" Aha ! " , zei de olifant , " een kikker met een kroon ! U bent vast een prins , uwe koninklijke kikkerheid . "']
    print 'tests passed'

if __name__ == '__main__':
    test()
        