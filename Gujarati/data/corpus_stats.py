## checking tokens per word to have an estimate how much dataset would be needed
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
##
import time
path=r"D:\Desktop\IIIT HYD\LMA\individual-project-Vidur-Shah04\Gujarati\data\raw\downloaded\IITB-IndicMonoDoc\gu\shard-4.txt"
doc_count=0
word_count=0
char_count=0
start=time.time()
# token_count=0
with open(path,"r",encoding="utf-8") as fh:
    for line in fh:
        if line.strip()=="<DOC_END>":
            doc_count+=1
        elif line.strip()=="<DOC_START>":continue
        else:
            # tokens = tokenizer.tokenize(line)
            words=line.strip().split()
            word_count+=len(words)
            # token_count+=len(tokens)
            for word in words:
                char_count+=len(word)
end=time.time()
print(f"file has {doc_count} docs with word count {word_count} and {char_count} characters")
# print(f"tokens per word is {token_count/word_count}")
print(f"the execution took {end-start} secs")
# from transformers import AutoTokenizer
# import time

# tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")

# path = r"D:\Desktop\IIIT HYD\LMA\individual-project-Vidur-Shah04\Gujarati\data\raw\downloaded\IITB-IndicMonoDoc\gu\shard-4.txt"

# word_count = 0
# token_count = 0
# doc_count = 0

# start = time.time()

# with open(path, "r", encoding="utf-8") as fh:
#     for line in fh:
#         stripped = line.strip()

#         if stripped == "<DOC_END>":
#             doc_count += 1
#             continue

#         if stripped == "<DOC_START>":
#             continue

#         words = stripped.split()

#         # Stop after ~100k words
#         if word_count + len(words) > 1_000_000:
#             words = words[:1_000_000 - word_count]

#         word_count += len(words)

#         text = " ".join(words)
#         tokens = tokenizer.tokenize(text)
#         token_count += len(tokens)

#         if word_count >= 1_000_000:
#             break

# end = time.time()

# print(f"Words: {word_count}")
# print(f"Tokens: {token_count}")
# print(f"Tokens per word: {token_count / word_count:.3f}")
# print(f"Time: {end - start:.2f} seconds")
##outputs
