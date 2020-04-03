
# coding: utf-8

# In[ ]:

def splicenum():
    data1= open('splice.data.txt', 'r')
    f= data1.readlines()
    for i in range(len(f)):
        f[i]= f[i].split(',')
    for i in range(len(f)):
        del(f[i][1])
    for i in range(len(f)):
        f[i][1]=f[i][1].split()[0]
    c=[]
    for i in range(len(f)):
        b=[]
        if f[i][0]=='EI':
            b.append(0)
        elif f[i][0]=='IE':
            b.append(1)
        elif f[i][0]=='N':
            b.append(2)
        a=[]
        for j in range(len(f[i][1])):
            if f[i][1][j]=='A':
                a.append(0)
            elif f[i][1][j]=='C':
                a.append(1)
            elif f[i][1][j]=='G':
                a.append(2)
            elif f[i][1][j]=='T':
                a.append(3)
            else:
                a.append(8)
        b.append(a)
        c.append(b)
    cdata=[]
    ctarg=[]

    for i in range(len(c)):
        cdata.append(c[i][1])
        ctarg.append((c[i][0]))
    return cdata , ctarg

