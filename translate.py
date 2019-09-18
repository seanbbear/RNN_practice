from opencc import OpenCC

def translate():
    cc = OpenCC('s2t')
    source = open('3000.txt', 'r', encoding = 'utf-8')
    result = open('3000_tw.txt', 'w', encoding = 'utf-8')
    #source就放純文字檔，轉完就放進去result
    count = 0
    while True:
        line = source.readline()
        line = cc.convert(line)
        if not line:  #readline會一直讀下去，這邊做的break
            break
        # print(line)
        count = count +1
        result.write(line) 
        print('===已處理'+str(count)+'行===')
    source.close()        
    result.close()

translate()