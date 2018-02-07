def J2H(mecab, text):

    mecab.parse("")
    node = mecab.parseToNode(text)
    yomi = ""
    while node:
        parseresult = node.feature.split(",")
        if node.surface in ["酒", "柿"]:
            yomi += node.surface
        else:
            try:
                if parseresult[7] != "*":
                    yomi += parseresult[7]
            except:
                yomi += node.surface
        node = node.next
    return yomi