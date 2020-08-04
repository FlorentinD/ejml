/*
 * Copyright (c) 2009-2020, Peter Abeles. All Rights Reserved.
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

package org.ejml.sparse.csc;

import org.ejml.MatrixDimensionException;
import org.ejml.data.DGrowArray;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.data.IGrowArray;
import org.ejml.ops.DSemiRing;
import org.ejml.sparse.csc.misc.ImplCommonOpsWithSemiRing_DSCC;
import org.ejml.sparse.csc.mult.ImplSparseSparseMultWithSemiRing_DSCC;

import javax.annotation.Nullable;

import static org.ejml.UtilEjml.reshapeOrDeclare;
import static org.ejml.UtilEjml.stringShapes;


public class CommonOpsWithSemiRing_DSCC {

    public static DMatrixSparseCSC mult(DMatrixSparseCSC A, DMatrixSparseCSC B, @Nullable DMatrixSparseCSC output, DSemiRing semiRing) {
        return mult(A, B, output, semiRing, null, null);
    }

    /**
     * Performs matrix multiplication.  output = A*B
     *
     * @param A  (Input) Matrix. Not modified.
     * @param B  (Input) Matrix. Not modified.
     * @param output  (Output) Storage for results.  Data length is increased if increased if insufficient.
     * @param gw (Optional) Storage for internal workspace.  Can be null.
     * @param gx (Optional) Storage for internal workspace.  Can be null.
     */
    public static DMatrixSparseCSC mult(DMatrixSparseCSC A, DMatrixSparseCSC B, @Nullable DMatrixSparseCSC output, DSemiRing semiRing,
                            @Nullable IGrowArray gw, @Nullable DGrowArray gx) {
        if (A.numCols != B.numRows)
            throw new MatrixDimensionException("Inconsistent matrix shapes. " + stringShapes(A, B));
        output = reshapeOrDeclare(output,A,A.numRows,B.numCols);

        ImplSparseSparseMultWithSemiRing_DSCC.mult(A, B, output, semiRing, gw, gx);

        return output;
    }

    public static DMatrixSparseCSC multTransA(DMatrixSparseCSC A, DMatrixSparseCSC B, DMatrixSparseCSC output, DSemiRing semiRing,
                                     @Nullable IGrowArray gw, @Nullable DGrowArray gx) {
        if (A.numRows != B.numRows)
            throw new MatrixDimensionException("Inconsistent matrix shapes. " + stringShapes(A, B));
        output = reshapeOrDeclare(output,A,A.numCols,B.numCols);

        ImplSparseSparseMultWithSemiRing_DSCC.multTransA(A, B, output, semiRing, gw, gx);

        return output;
    }

    /**
     * Performs matrix multiplication.  output = A*B<sup>T</sup>. B needs to be sorted and will be sorted if it
     * has not already been sorted.
     *
     * @param A  (Input) Matrix. Not modified.
     * @param B  (Input) Matrix. Value not modified but indicies will be sorted if not sorted already.
     * @param output  (Output) Storage for results.  Data length is increased if increased if insufficient.
     * @param gw (Optional) Storage for internal workspace.  Can be null.
     * @param gx (Optional) Storage for internal workspace.  Can be null.
     */
    public static DMatrixSparseCSC multTransB(DMatrixSparseCSC A, DMatrixSparseCSC B, @Nullable DMatrixSparseCSC output, DSemiRing semiRing,
                                  @Nullable IGrowArray gw, @Nullable DGrowArray gx) {
        if (A.numCols != B.numCols)
            throw new MatrixDimensionException("Inconsistent matrix shapes. " + stringShapes(A, B));
        output = reshapeOrDeclare(output, A, A.numRows, B.numRows);

        if (!B.isIndicesSorted())
            B.sortIndices(null);

        ImplSparseSparseMultWithSemiRing_DSCC.multTransB(A, B, output, semiRing, gw, gx);

        return output;
    }


    /**
     * Performs matrix multiplication.  output = A*B
     *
     * @param A Matrix
     * @param B Dense Matrix
     * @param output Dense Matrix
     */
    public static DMatrixRMaj mult(DMatrixSparseCSC A, DMatrixRMaj B, @Nullable DMatrixRMaj output, DSemiRing semiRing) {
        if (A.numCols != B.numRows)
            throw new MatrixDimensionException("Inconsistent matrix shapes. " + stringShapes(A, B));

        output = reshapeOrDeclare(output,A.numRows,B.numCols);

        ImplSparseSparseMultWithSemiRing_DSCC.mult(A, B, output, semiRing);

        return output;
    }

    /**
     * <p>output = output + A*B</p>
     */
    public static void multAdd(DMatrixSparseCSC A, DMatrixRMaj B, @Nullable DMatrixRMaj output, DSemiRing semiRing) {
        if (A.numRows != output.numRows || B.numCols != output.numCols)
            throw new IllegalArgumentException("Inconsistent matrix shapes. " + stringShapes(A, B, output));

        ImplSparseSparseMultWithSemiRing_DSCC.multAdd(A, B, output, semiRing);
    }

    /**
     * Performs matrix multiplication.  output = A<sup>T</sup>*B
     *
     * @param A Matrix
     * @param B Dense Matrix
     * @param output Dense Matrix
     */
    public static DMatrixRMaj multTransA(DMatrixSparseCSC A, DMatrixRMaj B, @Nullable DMatrixRMaj output, DSemiRing semiRing) {
        if (A.numRows != B.numRows)
            throw new MatrixDimensionException("Inconsistent matrix shapes. " + stringShapes(A, B));

        output = reshapeOrDeclare(output, A.numCols, B.numCols);

        ImplSparseSparseMultWithSemiRing_DSCC.multTransA(A, B, output, semiRing);

        return output;
    }

    /**
     * <p>output = output + A<sup>T</sup>*B</p>
     */
    public static void multAddTransA(DMatrixSparseCSC A, DMatrixRMaj B, DMatrixRMaj output, DSemiRing semiRing) {
        if (A.numCols != output.numRows || B.numCols != output.numCols)
            throw new IllegalArgumentException("Inconsistent matrix shapes. " + stringShapes(A, B, output));

        ImplSparseSparseMultWithSemiRing_DSCC.multAddTransA(A, B, output, semiRing);
    }

    /**
     * Performs matrix multiplication.  output = A*B<sup>T</sup>
     *
     * @param A Matrix
     * @param B Dense Matrix
     * @param output Dense Matrix
     */
    public static DMatrixRMaj multTransB(DMatrixSparseCSC A, DMatrixRMaj B, @Nullable DMatrixRMaj output, DSemiRing semiRing) {
        // todo: combine with multAdd as only difference is that output is filled with zero before
        if (A.numCols != B.numCols)
            throw new MatrixDimensionException("Inconsistent matrix shapes. " + stringShapes(A, B));
        output = reshapeOrDeclare(output, A.numRows, B.numRows);

        ImplSparseSparseMultWithSemiRing_DSCC.multTransB(A, B, output, semiRing);

        return output;
    }

    /**
     * <p>output = output + A*B<sup>T</sup></p>
     */
    public static void multAddTransB(DMatrixSparseCSC A, DMatrixRMaj B, DMatrixRMaj output, DSemiRing semiRing) {
        if (A.numRows != output.numRows || B.numRows != output.numCols)
            throw new IllegalArgumentException("Inconsistent matrix shapes. " + stringShapes(A, B, output));

        // TODO: ? this is basically the equivalent of graphblas mult with specified accumulator op
        ImplSparseSparseMultWithSemiRing_DSCC.multAddTransB(A, B, output, semiRing);
    }

    /**
     * Performs matrix multiplication.  output = A<sup>T</sup>*B<sup>T</sup>
     *
     * @param A Matrix
     * @param B Dense Matrix
     * @param output Dense Matrix
     */
    public static DMatrixRMaj multTransAB(DMatrixSparseCSC A, DMatrixRMaj B, DMatrixRMaj output, DSemiRing semiRing) {
        if (A.numRows != B.numCols)
            throw new MatrixDimensionException("Inconsistent matrix shapes. " + stringShapes(A, B));
        output = reshapeOrDeclare(output, A.numCols, B.numRows);

        ImplSparseSparseMultWithSemiRing_DSCC.multTransAB(A, B, output, semiRing);

        return output;
    }


    /**
     * <p>C = C + A<sup>T</sup>*B<sup>T</sup></p>
     */
    public static void multAddTransAB(DMatrixSparseCSC A, DMatrixRMaj B, DMatrixRMaj C, DSemiRing semiRing) {
        // TODO: this is basically the equivalent of graphblas mult with specified accumulator op
        if (A.numCols != C.numRows || B.numRows != C.numCols)
            throw new IllegalArgumentException("Inconsistent matrix shapes. " + stringShapes(A, B, C));

        ImplSparseSparseMultWithSemiRing_DSCC.multAddTransAB(A, B, C, semiRing);
    }

    /**
     * Performs matrix addition:<br>
     * C = &alpha;A + &beta;B
     *
     * @param alpha scalar value multiplied against A
     * @param A     Matrix
     * @param beta  scalar value multiplied against B
     * @param B     Matrix
     * @param C     Output matrix.
     * @param gw    (Optional) Storage for internal workspace.  Can be null.
     * @param gx    (Optional) Storage for internal workspace.  Can be null.
     */
    public static void add(double alpha, DMatrixSparseCSC A, double beta, DMatrixSparseCSC B, DMatrixSparseCSC C, DSemiRing semiRing,
                           @Nullable IGrowArray gw, @Nullable DGrowArray gx) {
        if (A.numRows != B.numRows || A.numCols != B.numCols)
            throw new MatrixDimensionException("Inconsistent matrix shapes. " + stringShapes(A, B));
        C.reshape(A.numRows, A.numCols);

        ImplCommonOpsWithSemiRing_DSCC.add(alpha, A, beta, B, C, semiRing, gw, gx);
    }

    /**
     * Performs an element-wise multiplication.<br>
     * C[i,j] = A[i,j]*B[i,j]<br>
     * All matrices must have the same shape.
     *
     * @param A  (Input) Matrix.
     * @param B  (Input) Matrix
     * @param C  (Output) Matrix. data array is grown to min(A.nz_length,B.nz_length), resulting a in a large speed boost.
     * @param gw (Optional) Storage for internal workspace.  Can be null.
     * @param gx (Optional) Storage for internal workspace.  Can be null.
     */
    public static void elementMult(DMatrixSparseCSC A, DMatrixSparseCSC B, DMatrixSparseCSC C, DSemiRing semiRing,
                                   @Nullable IGrowArray gw, @Nullable DGrowArray gx) {
        if (A.numCols != B.numCols || A.numRows != B.numRows)
            throw new MatrixDimensionException("All inputs must have the same number of rows and columns. " + stringShapes(A, B));
        C.reshape(A.numRows, A.numCols);

        ImplCommonOpsWithSemiRing_DSCC.elementMult(A, B, C, semiRing, gw, gx);
    }
}

