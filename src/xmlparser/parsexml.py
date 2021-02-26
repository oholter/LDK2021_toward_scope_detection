from xml.dom.minidom import parse
import xml.dom.minidom
from nltk import sent_tokenize


def print_reqs(requirements, file=None, filter_shall=True):
    num_req_sents = 0
    num_shall_reqs = 0
    num_reqs = len(requirements)
    print("sec\treq\tsent_num\tsent")


    for req in requirements:
        req_num = req.getAttribute('num')
        sub1 = req.parentNode
        part = sub1.parentNode
        section = part.parentNode
        # print(req.firstChild.nodeValue)
        try:
            sents = sent_tokenize(req.firstChild.nodeValue)
        except AttributeError:
            sents = []
        try:
            section_num = section.getAttribute('num')
        except AttributeError:
            section_num = 0
        part_num = part.getAttribute('num')
        sub1_num = sub1.getAttribute('num')
        for i, sent in enumerate(sents):
            sent = sent.replace('\n', ' ')
            print("{}\t{}\t{}\t{}".format(section_num, req_num, i + 1, sent.strip()))
            num_req_sents +=1
            if ' shall ' in sent:
                num_shall_reqs += 1


    if file:
        with open(file, 'w') as F:
            F.write("sec\treq\tsent_num\tsent\n")
            for req in requirements:
                req_num = req.getAttribute('num')
                sub1 = req.parentNode
                part = sub1.parentNode
                section = part.parentNode
                try:
                    sents = sent_tokenize(req.firstChild.nodeValue)
                except AttributeError:
                    sents = []
                try:
                    section_num = section.getAttribute('num')
                except AttributeError:
                    section_num = 0
                for i, sent in enumerate(sents):
                    if not filter_shall:
                        sent = sent.replace('\n', ' ')
                        F.write("{}\t{}\t{}\t{}\n".format(section_num, req_num, i + 1, sent.strip()))
                    else:
                        sent = sent.replace('\n', ' ')
                        if ' shall ' in sent:
                            F.write("{}\t{}\t{}\t{}\n".format(section_num, req_num, i + 1, sent.strip()))



    print("\ntotal number of reqs: {}".format(num_reqs))
    print("total number of req sents: {}".format(num_req_sents))
    print("number of shall req sents: {}".format(num_shall_reqs))


def print_sub(subs, filter_shall=False):
    num_sents = 0
    num_shall_sents = 0
    num_subs = len(subs)
    print("sec\tsub\tsent_num\tsent")

    for sub in subs:
        num = sub.getAttribute('num')
        part = sub.parentNode
        section = part.parentNode
        sents = sent_tokenize(sub.firstChild.nodeValue)

        #section_num = section.getAttribute('num')
        section_num = num[0]
        for i, sent in enumerate(sents):
            num_sents += 1
            if not filter_shall:
                sent = sent.replace('\n', ' ')
                print("{}\t{}\t{}\t{}".format(section_num, num, i + 1, sent.strip()))
                if ' shall ' in sent:
                    num_shall_sents += 1

            else:
                if ' shall ' in sent:
                    num_shall_sents += 1
                    sent = sent.replace('\n', ' ')
                    print("{}\t{}\t{}\t{}".format(section_num, num, i + 1, sent.strip()))




    print("\ntotal number of subs: {}".format(num_subs))
    print("total number of sents: {}".format(num_sents))
    print("number of shall sents: {}".format(num_shall_sents))



if __name__ == "__main__":
    path = "xml/DNVGL-RU-FD.xml"
    tsv_path = "tsv/DNVGL-RU-FD.tsv"

    DOMTree = xml.dom.minidom.parse(path)
    collection = DOMTree.documentElement
    if collection.hasAttribute('document'):
        print('root element has: '.format(collection.getAttribute('document')))

    requirements = collection.getElementsByTagName("req")
    sub2 = collection.getElementsByTagName("sub2")
    sub1 = collection.getElementsByTagName("sub1")
    part = collection.getElementsByTagName("part")
    sec = collection.getElementsByTagName("section")


    #print_reqs(requirements, file=tsv_path, filter_shall=True)
    #print_sub(sub1, filter_shall=True)
    print_sub(part, filter_shall=True)

    # not always used on ru-ship
    #print_sub(sub2, filter_shall=True)
