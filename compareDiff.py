# fileOld = 'questions_VP.txt'
# fileNew = 'questions_Who.txt'

# diffLines = []

# with open(fileOld, 'r') as f_old:
#     line_old = f_old.readlines()

# with open(fileNew, 'r') as f_new:
#     line_new = f_new.readlines()

# i_old = 0
# i_new = 0

# while i_new < len(line_new):
#     index_new = int(line_new[i_new].split()[0])
#     if i_old < len(line_old):

#         index_old = int(line_old[i_old].split()[0])
        
#         if index_new < index_old:
#             diffLines.append('NEW\n')
#             diffLines.append(line_new[i_new]+'\n')
#             i_new += 1
#         elif index_new > index_old:
#             diffLines.append('OLD\n')
#             diffLines.append(line_old[i_old] + '\n')
#             i_old += 1        
#         else:
#             if line_old[i_old] != line_new[i_new]:
#                 diffLines.append(line_old[i_old] + line_new[i_new] + '\n')
#             i_new += 1
#             i_old += 1
#     else:
#         diffLines.append('NEW\n')
#         diffLines.append(line_new[i_new]+'\n')
#         i_new += 1

    

# with open('debug', 'w') as f:
#     for line in diffLines:
#         f.write(line)
#         f.write('\n')

with open('debug', 'r') as f1:
    s1 = f1.readlines()
with open('train.txt', 'r') as f2:
    s2 = f2.readlines()

for i in range(len(s2)):
    if s1[i].strip()[:-1].strip() != s2[i].strip()[:-1].strip():
        print(s1[i].strip()[:-1])
        print(s2[i].strip()[:-1].strip())
        print()
