package numgo

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
	ut "test_ai/utils"
)

func NewDataLoader(d *dataSet,batchSize int,shuffle bool) *DataLoader {
	maxIter:=d.Len()/batchSize
	dr := &DataLoader{dataSet: d, batchSize: batchSize, shuffle: shuffle, maxIter: maxIter}
	dr.reset()
	return dr
}

type DataLoader struct {
	dataSet   *dataSet
	batchSize int
	dataSize  int
	gpu       bool
	shuffle   bool
	maxIter   int
	iteration int
	index     []int
}

func (d *DataLoader) reset() {
	d.iteration=0
	if d.shuffle{
		d.index=rand.Perm(d.dataSet.Len())
	}else{
		d.index=ut.ArangeInt(0,d.dataSet.Len(),1)
	}
}
func (d *DataLoader) HasNext() bool {
	return d.iteration<d.maxIter
}
func (d *DataLoader) Next() (x *mat.Dense,t []int){
	i,batchSize:= d.iteration,d.batchSize
	batchIndex:=d.index[i*batchSize:(i+1)*batchSize]
	x=SelRow(d.dataSet.Data,batchIndex...)
	//todo change
	t=ut.SelRowInt(d.dataSet.Label,batchIndex...)
	return x,t
}
func (d *DataLoader) ToGpu() {
	d.gpu=true
}
func (d *DataLoader) ToCpu() {
	d.gpu=false
}
