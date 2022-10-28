import argparse
import io
import os
import spacy
import scripts.align_text_korean as align_text
from scripts.extractPos.align import align_pos_with_exceptions
import hunspell
import re
import tqdm

# The input files may be tokenized or untokenized.
# Assumption 1: Each line in each file aligns exactly.
# Assumption 2: Each line in each file is at least 1 sentence in orig and cor.
def main(args):
    try:
        hobj = hunspell.HunSpell(args.hunspell + '/ko.dic', args.hunspell + '/ko.aff')
    except AttributeError: # When you use cyhunspell
        hobj = hunspell.Hunspell('ko', hunspell_data_dir=args.hunspell)
    pat = re.compile(r"([-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》])")
    basename = os.path.dirname(os.path.realpath(__file__))
    editcount, linecount = 0, 0
    num_edits = 0
    print("Loading resources...")
    # Load Tokenizer and other resources
    # Setup output m2 file based on corrected file name.
    m2_out = open(args.out if args.out.endswith(".m2") else args.out+".m2", "w")
    with open(args.orig, 'r') as f:
        orig = f.read().split("\n")
    with open(args.cor, 'r') as f:
        cor = f.read().split("\n")
    print("Processing files...")
    i = 0
    # Process each pre-aligned sentence pair.
    for idx, (orig_sent, cor_sent) in enumerate(zip(orig, cor)):
        i += 1
        if i % 1000 == 0:
            print("Processing: ", i)
        # Ignore empty sentences
        if not orig_sent and not cor_sent: continue
        # IMPORTANT: ADD spaces between punctuations. -> moved to the dataset preprocessing part.
        #orig_sent = re.sub(' +', ' ', pat.sub(" \\1 ", orig_sent)).strip()
        #cor_sent = re.sub(' +', ' ', pat.sub(" \\1 ", cor_sent)).strip()

        # Get a list of string toks for each.
        orig_toks = orig_sent.split(" ")
        cor_toks = cor_sent.split(" ")
        orig_extra, cor_extra = align_pos_with_exceptions(orig_sent, cor_sent)
        # FIXED ALIGN_POS

        # Auto align the sentence and extract the edits.
        auto_edits = align_text.getAutoAlignedEdits(orig_toks, cor_toks, orig_extra, cor_extra,
                                                    args.merge, hobj, verbose=not args.noprint, verbose_unclassified=args.verbose_unclassified)
        num_edits += len(auto_edits)

        if len(args.logset) != 0: # filter auto_edits.
            filtered_edits = list(filter(lambda x: (x[2] in args.logset), auto_edits))
            editcount += len(filtered_edits)
        if len(args.logset) == 0 or len(filtered_edits) > 0:
            linecount += 1
            # Write orig_toks to output.
            m2_out.write("S "+" ".join(orig_toks)+"\n")
            # If there are no edits, write an explicit dummy edit.
            if not auto_edits:
                m2_out.write("A -1 -1|||noop||||||REQUIRED|||-NONE-|||0\n")
            # Write the auto edits to the file.
            for auto_edit in auto_edits:
                # Write the edit to output.
                m2_out.write(formatEdit(auto_edit)+"\n")
            # Write new line after each sentence.
            m2_out.write("\n")
            if not args.noprint:
                print(f"Index: {idx}, Count: {linecount}")
                print("S "+" ".join(orig_toks))
                if not auto_edits:
                    print("A -1 -1|||noop||||||REQUIRED|||-NONE-|||0")
                    # Write the auto edits to the file.
                for auto_edit in auto_edits:
                    # Write the edit to output.
                    print(formatEdit(auto_edit))
                    # Write new line after each sentence.
                print("\n")
    #total_statistics = f"Total lines: {idx + 1}, count: {linecount} ({round((linecount * 100)/(idx + 1), 3)} %)\nTotal edits: {num_edits} count: {editcount} ({round((editcount * 100)/(num_edits), 3)} %)"
    #print(total_statistics)

# Function to format an edit into M2 output format.
def formatEdit(edit, coder_id=0):
    # edit = [start, end, cat, cor]
    span = " ".join(["A", str(edit[0]), str(edit[1])])
    return "|||".join([span, edit[2], edit[3], "REQUIRED", "-NONE-", str(coder_id)])


if __name__ == "__main__":
    # Define and parse program input
    parser = argparse.ArgumentParser(description="Convert parallel original and corrected text files (1 sentence per line) into M2 format.\nThe default uses Damerau-Levenshtein and merging rules and assumes tokenized text.",
                                        formatter_class=argparse.RawTextHelpFormatter,
                                        usage="%(prog)s [-h] [options] -orig ORIG -cor COR -out OUT")
    parser.add_argument("-orig",
                        help="The path to the original text file.",
                        required=True)
    parser.add_argument("-cor",
                        help="The path to the corrected text file.",
                        required=True)
    parser.add_argument("-out",
                        help="The full filename of where you want the output m2 file saved.")
    parser.add_argument("-merge",
                        help="Choose a merging strategy for an automatic alignment.\n"
                                "all-split: Merge nothing; e.g. MSSDI -> M, S, S, D, I\n"
                                "all-merge: Merge adjacent non-matches; e.g. MSSDI -> M, SSDI\n"
                                "all-equal: Merge adjacent same-type non-matches; e.g. MSSDI -> M, SS, D, I\n"
                                "rules: Use our own rule-based merging strategy (default)",
                        default="rules")
    parser.add_argument("-logset", help="Whether to output m2 file for a specific type of log - default is ''",
                        nargs='*', default=[])
    parser.add_argument("-noprint", action="store_true", help="Whether to print it on the log.")
    parser.add_argument("-verbose_unclassified", action="store_true", help="Whether to save all POS tags for the error type UNCLASSIFIED.")
    parser.add_argument("-hunspell", default="aff-dic/", help="The directory where Hunspell korean library is saved")
    args = parser.parse_args()
    # Run the program.
    main(args)
