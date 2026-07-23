import numpy as np

def elem_stiff(struct):

    b = float(struct["b"])
    h = float(struct["h"])
    nelemx = float(struct["nelemx"])
    nelemy = float(struct["nelemy"])
    E = float(struct["E"])
    nu = float(struct["v"])
    E1 = E / (1 - nu**2)
    be2 = 0.5 * (b / nelemx)
    he2 = 0.5 * (h / nelemy)
    tck = float(struct["e"])
    E1tck = E1 * tck
    abe = be2 * he2 * tck

    kel0_11 = abe*(E1/(3*be2**2) - (E1*(nu - 1))/(6*he2**2))
    kel0_21 = (E1tck*(nu + 1))/8
    kel0_31 = -abe*(E1/(3*be2**2) + (E1*(nu - 1))/(12*he2**2))
    kel0_41 = (E1tck*(3*nu - 1))/8
    kel0_51 = -abe*(E1/(6*be2**2) - (E1*(nu - 1))/(12*he2**2))
    kel0_61 = -(E1tck*(nu + 1))/8
    kel0_71 = abe*(E1/(6*be2**2) + (E1*(nu - 1))/(6*he2**2))
    kel0_81 = -(E1tck*(3*nu - 1))/8

    kel0_12 = kel0_21
    kel0_22 = abe*(E1/(3*he2**2) - (E1*(nu - 1))/(6*be2**2))
    kel0_32 = -(E1tck*(3*nu - 1))/8
    kel0_42 = abe*(E1/(6*he2**2) + (E1*(nu - 1))/(6*be2**2))
    kel0_52 = -(E1tck*(nu + 1))/8
    kel0_62 = -abe*(E1/(6*he2**2) - (E1*(nu - 1))/(12*be2**2))
    kel0_72 = (E1tck*(3*nu - 1))/8
    kel0_82 = -abe*(E1/(3*he2**2) + (E1*(nu - 1))/(12*be2**2))

    kel0_13 = kel0_31
    kel0_23 = kel0_32
    kel0_33 = abe*(E1/(3*be2**2) - (E1*(nu - 1))/(6*he2**2))
    kel0_43 = -(E1tck*(nu + 1))/8
    kel0_53 = abe*(E1/(6*be2**2) + (E1*(nu - 1))/(6*he2**2))
    kel0_63 = (E1tck*(3*nu - 1))/8
    kel0_73 = -abe*(E1/(6*be2**2) - (E1*(nu - 1))/(12*he2**2))
    kel0_83 = (E1tck*(nu + 1))/8

    kel0_14 = kel0_41
    kel0_24 = kel0_42
    kel0_34 = kel0_43
    kel0_44 = abe*(E1/(3*he2**2) - (E1*(nu - 1))/(6*be2**2))
    kel0_54 = -(E1tck*(3*nu - 1))/8
    kel0_64 = -abe*(E1/(3*he2**2) + (E1*(nu - 1))/(12*be2**2))
    kel0_74 = (E1tck*(nu + 1))/8
    kel0_84 = -abe*(E1/(6*he2**2) - (E1*(nu - 1))/(12*be2**2))

    kel0_15 = kel0_51
    kel0_25 = kel0_52
    kel0_35 = kel0_53
    kel0_45 = kel0_54
    kel0_55 = abe*(E1/(3*be2**2) - (E1*(nu - 1))/(6*he2**2))
    kel0_65 = (E1tck*(nu + 1))/8
    kel0_75 = -abe*(E1/(3*be2**2) + (E1*(nu - 1))/(12*he2**2))
    kel0_85 = (E1tck*(3*nu - 1))/8

    kel0_16 = kel0_61
    kel0_26 = kel0_62
    kel0_36 = kel0_63
    kel0_46 = kel0_64
    kel0_56 = kel0_65
    kel0_66 = abe*(E1/(3*he2**2) - (E1*(nu - 1))/(6*be2**2))
    kel0_76 = -(E1tck*(3*nu - 1))/8
    kel0_86 = abe*(E1/(6*he2**2) + (E1*(nu - 1))/(6*be2**2))

    kel0_17 = kel0_71
    kel0_27 = kel0_72
    kel0_37 = kel0_73
    kel0_47 = kel0_74
    kel0_57 = kel0_75
    kel0_67 = kel0_76
    kel0_77 = abe*(E1/(3*be2**2) - (E1*(nu - 1))/(6*he2**2))
    kel0_87 = -(E1tck*(nu + 1))/8

    kel0_18 = kel0_81
    kel0_28 = kel0_82
    kel0_38 = kel0_83
    kel0_48 = kel0_84
    kel0_58 = kel0_85
    kel0_68 = kel0_86
    kel0_78 = kel0_87
    kel0_88 = abe*(E1/(3*he2**2) - (E1*(nu - 1))/(6*be2**2))

    kel0 = np.array([
        [kel0_11, kel0_12, kel0_13, kel0_14, kel0_15, kel0_16, kel0_17, kel0_18],
        [kel0_21, kel0_22, kel0_23, kel0_24, kel0_25, kel0_26, kel0_27, kel0_28],
        [kel0_31, kel0_32, kel0_33, kel0_34, kel0_35, kel0_36, kel0_37, kel0_38],
        [kel0_41, kel0_42, kel0_43, kel0_44, kel0_45, kel0_46, kel0_47, kel0_48],
        [kel0_51, kel0_52, kel0_53, kel0_54, kel0_55, kel0_56, kel0_57, kel0_58],
        [kel0_61, kel0_62, kel0_63, kel0_64, kel0_65, kel0_66, kel0_67, kel0_68],
        [kel0_71, kel0_72, kel0_73, kel0_74, kel0_75, kel0_76, kel0_77, kel0_78],
        [kel0_81, kel0_82, kel0_83, kel0_84, kel0_85, kel0_86, kel0_87, kel0_88]
    ], dtype= np.double)

    return kel0