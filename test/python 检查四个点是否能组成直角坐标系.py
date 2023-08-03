def is_cartesian(p1, p2, p3, p4):
    p1p2=p1-p2
    p1p3=p1-p3
    p1p4=p1-p4
    p2p3=p2-p3
    p2p4=p2-p4
    p3p4=p3-p4
    lines=[p1p2,p1p3,p1p4,p2p3,p2p4,p3p4]
