package numgo

import (
	"math"
	"math/rand"

	nt "test_ai/tensor"
	ut "test_ai/utils"
)

type IDataset interface {
	prePare(data *DataSet)
}
type DataSet struct {
	Train       bool
	Data        *nt.Tensor
	Label       []float64
	IDataset    IDataset
	trans       *Compose
	targetTrans *Compose
}

func (d *DataSet) Init() {
	d.IDataset.prePare(d)
}
func (d *DataSet) Len() int {
	return len(d.Label)
}
func (d *DataSet) Get(index int) ([]float64, float64) {
	data := d.Data.Slices(0, index).Data()
	////todo debug image show
	//pv := nd.NewVec(data...)
	//if pv.Sum(nil,true).Var()>1{
	//	PrintImg(data,d.Label[index])
	//}
	if d.trans != nil {
		d.trans.Run(data)
	}
	if d.Label == nil {
		return data, 0
	} else {
		label := d.Label[index]
		return data, label
	}
}

type spiral struct {
	DataSet
}

func (s spiral) prePare(d *DataSet) {
	data, label := GetSpiral(s.Train)
	dr, dc, dd := ut.Flatten(data)
	d.Data = nt.NewData(dd, dr, dc)
	d.Label = label
}

func Spiral(train bool) *DataSet {
	d := DataSet{IDataset: &spiral{DataSet{Train: train}}}
	d.Init()
	return &d
}

func GetSpiral(train bool) ([][]float64, []float64) {
	if train {
		rand.Seed(1984)
	} else {
		rand.Seed(2020)
	}
	numData, numClass := 100.0, 3.0
	dataSize := int(numData * numClass)
	x := make([][]float64, dataSize)
	t := make([]float64, dataSize)
	for j := 0.0; j < numClass; j++ {
		for i := 0.0; i < numData; i++ {
			rate := i / numData
			radius := 1.0 * rate
			theta := j*4.0 + 4.0*rate + rand.NormFloat64()*0.2
			ix := int(numData*j + i)
			xv := []float64{radius * math.Sin(theta), radius * math.Cos(theta)}
			x[ix] = xv
			t[ix] = j //分类标签
		}
	}
	//shuffle
	rand.Shuffle(len(x), func(i, j int) {
		x[i], x[j] = x[j], x[i]
		t[i], t[j] = t[j], t[i]
	})
	return x, t
}

//
type sinCurve struct {
	DataSet
}

func (s *sinCurve) prePare(d *DataSet) {
	numData := 1000
	x := ut.Linspace(0, 2*math.Pi, numData)
	noise := ut.RandUniformFloats(-0.05, 0.05, len(x))
	xv := nt.NewVec(x...)
	nv := nt.NewVec(noise...)
	y := &nt.Tensor{}
	if s.Train {
		y = nt.Add(nt.Sin(xv), nv)
	} else {
		y = nt.Cos(xv)
	}

	yd := y.Data()
	d.Data = nt.NewVec(yd[:len(yd)-2]...)

	//取下一个作为预测点
	d.Label = yd[1 : len(yd)-1]
}

func SinCurve(train bool) *DataSet {
	d := DataSet{IDataset: &sinCurve{DataSet{Train: train}}}
	d.Init()
	return &d
}
