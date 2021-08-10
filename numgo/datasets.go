package numgo

import (
	"math"
	"math/rand"

	nd "test_ai/numed"
	ut "test_ai/utils"
)

type IDataset interface {
	prePare(data *DataSet)
}
type DataSet struct {
	Train       bool
	Data        *nd.NumEd
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
	data := d.Data.RowData(index)
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
	d.Data = nd.NewDense(ut.Flatten(data))
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
	xv := nd.NewVec(x...)
	nv := nd.NewVec(noise...)
	y := &nd.NumEd{}
	if s.Train {
		y = nd.Add(xv.Sin(), nv)
	} else {
		y = xv.Cos()
	}

	t := y.Slice(0, 1, 0, len(x)-1)
	sp := t.Shape()
	d.Data = t.Reshape(sp.C, sp.R)

	//yt 取下一个点，来作为标签，判断模型输入x以后输出的y是否达到标签的预期，就是sin的下一个点
	ys := y.Shape()
	yt := y.Reshape(ys.C, ys.R).ColData(0)[1:]
	d.Label = yt
}

func SinCurve(train bool) *DataSet {
	d := DataSet{IDataset: &sinCurve{DataSet{Train: train}}}
	d.Init()
	return &d
}
