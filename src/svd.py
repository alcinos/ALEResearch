import numpy as np


mat = np.zeros([32,1025*18])
count = 0
for i in range(4):
    for k in range(8):
        j=((k*4))
        f = open("results/bank_heist_RAM_d"+str(i)+"_m"+str(j)+"/weights_bank_heist_RAM_d"+str(i)+"_m"+str(j)+".w")
        for line in f:
            data=line.split()
            if(len(data)==3):
                mat[count][1025*int(data[0])+int(data[1])]=float(data[2])

        count+=1

u,s,v = np.linalg.svd(mat,full_matrices=1)

for i in range(len(s)):
    sigma = np.zeros((32,1025*18))
    sigma[:i+1,:i+1]=np.diag(s[:i+1])
    res = np.dot(u,sigma)
    res = np.dot(res,v)
    for j in range(32):
        f = open("svd_decomp/bank_heist/m_"+str(j)+"_"+str(i)+".w","w")
        f.write("18 1025\n")
        for k in range(18):
            for l in range(1025):
                if abs(res[j][k*1025+l])>1e-6:
                    f.write(str(k)+" "+str(l)+" "+str(res[j][k*1025+l])+"\n")
        f.close()
                    
