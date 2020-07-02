/*
 * Copyright (c) 2009-2019, Peter Abeles. All Rights Reserved.
 *
 * This file is part of Efficient Java Matrix Library (EJML).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ejml.example;

import org.ejml.data.*;
import org.ejml.ops.ConvertDMatrixStruct;
import org.ejml.sparse.csc.CommonOps_DSCC;

/**
 * Example showing how to construct and multiply sparse matrices
 *
 * @author Peter Abeles
 */
public class ExampleSparseMatrix2 {

    private static final int VERTEX_COUNT = 10;

    public static void main(String[] args) {
        DMatrixSparseTriplet work = new DMatrixSparseTriplet(VERTEX_COUNT, VERTEX_COUNT, 5);
        work.addItem(0, 1, 1);
        work.addItem(1, 2, 1);
        work.addItem(4, 5, 1);
        work.addItem(2, 3, 1);

        // convert into a CSC format
        DMatrixSparseCSC adjList = ConvertDMatrixStruct.convert(work, (DMatrixSparseCSC) null);

        DMatrixSparseCSC sourceVertexVector = new DMatrixSparseCSC(1, VERTEX_COUNT, 10);
        CommonOps_DSCC.fill(sourceVertexVector, 1);

        DMatrixSparseCSC result = new DMatrixSparseCSC(1, VERTEX_COUNT, 0);
        // needs defined semi-ring in order to work
        CommonOps_DSCC.mult(sourceVertexVector, adjList, result);
        result.printNonZero();
    }
}
