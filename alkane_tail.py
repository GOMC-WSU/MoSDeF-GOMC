__author__ = 'sallai'
from mbuild.compound import *
from mbuild.port import *

class AlkaneTail(Compound):

    @classmethod
    def create(cls, ctx={}):
        m = super(AlkaneTail, cls).create(ctx=ctx)
        m.add(HB((1, 0, 0)),'h1')
        m.add(HB((0, 1, 0)),'h2')
        m.add(HB((-1, 0, 0)),'h3')
        m.add(CB((0, 0, 0)),'c')

        m.add(Port.create(), 'female_port')
        m.female_port.transform(Translation((0,-0.7,0)))

        m.add(Port.create(), 'male_port')
        m.male_port.transform(RotationAroundZ(pi))
        m.male_port.transform(Translation((0,-0.7,0)))

        return m
