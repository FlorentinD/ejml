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
import org.ejml.data.*;
import org.ejml.masks.Mask;
import org.ejml.ops.DBinaryOperator;
import org.ejml.ops.DSemiRing;
import org.ejml.sparse.csc.misc.ImplCommonOpsWithSemiRing_DSCC;
import org.ejml.sparse.csc.mult.ImplSparseSparseMultWithSemiRing_DSCC;
import org.jetbrains.annotations.Nullable;

import static org.ejml.UtilEjml.*;
import static org.ejml.sparse.csc.MaskUtil_DSCC.combineOutputs;


public class CommonOpsWithSemiRing_DSCC {

    public static DMatrixSparseCSC mult(DMatrixSparseCSC A, DMatrixSparseCSC B, @Nullable DMatrixSparseCSC output, DSemiRing semiRing,
                                        @Nullable Mask mask, @Nullable DBinaryOperator accumulator, boolean replaceOutput) {
        return mult(A, B, output, semiRing, mask, accumulator, replaceOutput, null, null);
    }

    public static DMatrixSparseCSC mult(DMatrixSparseCSC A, DMatrixSparseCSC B, @Nullable DMatrixSparseCSC output, DSemiRing semiRing) {
        return mult(A, B, output, semiRing, null, null, true, null, null);
    }

    /**
     * Performs matrix multiplication.  output = A*B
     *
     * @param A  (Input) Matrix. Not modified.
     * @param B  (Input) Matrix. Not modified.
     * @param output  (Output) Storage for results. Data length is increased if increased if insufficient.
     * @param semiRing Semi-Ring to define + and *
     * @param mask Mask for specifying which entries should be overwritten
     * @param accumulator Operator to combine result with existing entries in output matrix
     * @param replaceOutput If true, the value of the output parameter will be overwritten, otherwise they will be merged
     * @param gw       (Optional) Storage for internal workspace.  Can be null.
     * @param gx       (Optional) Storage for internal workspace.  Can be null.
     */
    public static DMatrixSparseCSC mult(DMatrixSparseCSC A, DMatrixSparseCSC B, @Nullable DMatrixSparseCSC output, DSemiRing semiRing,
                                        @Nullable Mask mask, @Nullable DBinaryOperator accumulator, boolean replaceOutput,
                                        @Nullable IGrowArray gw, @Nullable DGrowArray gx) {
        if (A.numCols != B.numRows)
            throw new MatrixDimensionException("Inconsistent matrix shapes. " + stringShapes(A, B));

        // !! important to do before reshape
        DMatrixSparseCSC initialOutput = MaskUtil_DSCC.maybeCacheInitialOutput(output, replaceOutput);

        output = reshapeOrDeclare(output, A, A.numRows, B.numCols);
        if (mask != null) {
            mask.compatible(output);
        }

        ImplSparseSparseMultWithSemiRing_DSCC.mult(A, B, output, semiRing, mask, gw, gx);

        return combineOutputs(output, initialOutput, null, accumulator);
    }

    /**
     * Performs matrix multiplication.  output = A*B
     *
     * @param A        Matrix
     * @param B        Dense Matrix
     * @param semiRing Semi-Ring to define + and *
     * @param output   Dense Matrix
     */
    public static DMatrixRMaj mult(DMatrixSparseCSC A, DMatrixRMaj B, @Nullable DMatrixRMaj output, DSemiRing semiRing) {
        if (A.numCols != B.numRows)
            throw new MatrixDimensionException("Inconsistent matrix shapes. " + stringShapes(A, B));

        output = reshapeOrDeclare(output, A.numRows, B.numCols);

        ImplSparseSparseMultWithSemiRing_DSCC.mult(A, B, output, semiRing);

        return output;
    }

    /**
     * <p>output = output + A*B</p>
     */
    public static void multAdd(DMatrixSparseCSC A, DMatrixRMaj B, DMatrixRMaj output, DSemiRing semiRing) {
        if (A.numRows != output.numRows || B.numCols != output.numCols)
            throw new IllegalArgumentException("Inconsistent matrix shapes. " + stringShapes(A, B, output));

        ImplSparseSparseMultWithSemiRing_DSCC.multAdd(A, B, output, semiRing);
    }

    /**
     * Performs matrix multiplication.  output = A<sup>T</sup>*B
     *
     * @param A        Matrix
     * @param B        Dense Matrix
     * @param output   Dense Matrix
     * @param semiRing Semi-Ring to define + and *
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
     * @param A        Matrix
     * @param B        Dense Matrix
     * @param output   Dense Matrix
     * @param semiRing Semi-Ring to define + and *
     */
    public static DMatrixRMaj multTransB(DMatrixSparseCSC A, DMatrixRMaj B, @Nullable DMatrixRMaj output, DSemiRing semiRing) {
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

        ImplSparseSparseMultWithSemiRing_DSCC.multAddTransB(A, B, output, semiRing);
    }

    /**
     * Performs matrix multiplication.  output = A<sup>T</sup>*B<sup>T</sup>
     *
     * @param A        Matrix
     * @param B        Dense Matrix
     * @param output   Dense Matrix
     * @param semiRing Semi-Ring to define + and *
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
        if (A.numCols != C.numRows || B.numRows != C.numCols)
            throw new IllegalArgumentException("Inconsistent matrix shapes. " + stringShapes(A, B, C));

        ImplSparseSparseMultWithSemiRing_DSCC.multAddTransAB(A, B, C, semiRing);
    }

    /**
     * Performs matrix addition:<br>
     * output = A + B
     *
     * @param A           Matrix
     * @param B           Matrix
     * @param output      (Optional)    Output matrix.
     * @param mask        Mask for specifying which entries should be overwritten
     * @param semiRing Semi-Ring to define + and *
     * @param accumulator Operator to combine result with existing entries in output matrix
     * @param replaceOutput If true, the value of the output parameter will be overwritten, otherwise they will be merged
     * @param gw          (Optional) Storage for internal workspace.  Can be null.
     * @param gx          (Optional) Storage for internal workspace.  Can be null.
     */
    public static DMatrixSparseCSC add(DMatrixSparseCSC A, DMatrixSparseCSC B, @Nullable DMatrixSparseCSC output, DSemiRing semiRing,
                                       @Nullable Mask mask, @Nullable DBinaryOperator accumulator, boolean replaceOutput,
                                       @Nullable IGrowArray gw, @Nullable DGrowArray gx) {
        if (A.numRows != B.numRows || A.numCols != B.numCols)
            throw new MatrixDimensionException("Inconsistent matrix shapes. " + stringShapes(A, B));
        DMatrixSparseCSC initialOutput = MaskUtil_DSCC.maybeCacheInitialOutput(output, replaceOutput);
        output = reshapeOrDeclare(output, A, A.numRows, A.numCols);
        if (mask != null) {
            mask.compatible(output);
        }

        ImplCommonOpsWithSemiRing_DSCC.add(A, B, output, semiRing.add.func, mask, gw, gx);

        return combineOutputs(output, initialOutput, null, accumulator);
    }

    /**
     * Performs in-place vector addition:<br>
     * u += v
     *
     * @param u           Matrix
     * @param v           Matrix
     * @param add         Binary operator to define `+`
     * @param gw          (Optional) Storage for internal workspace.  Can be null.
     */
    public static DVectorSparse add( DVectorSparse u, DVectorSparse v, DBinaryOperator add, @Nullable IGrowArray gw) {
        // no mask, as inPlace is not possible if entries of u should not be computed
        // no accumulator .. as inPlace add

        if (u.size() != v.size())
            throw new MatrixDimensionException("Inconsistent vector shapes. " + stringShapes(u, v));

        u.setIndicesSorted(false);

        int[] w = adjust(gw, u.size());

        // mapping vector-index -> nz-index
        for (int i = 0; i < u.nz_length(); i++) {
            w[u.nz_indices()[i]] = i + 1;
        }

        for (int i = 0; i < v.nz_length(); i++) {
            int vIndex = v.nz_indices()[i];
            double v_value = v.nz_values()[i];
            int u_nz_index = w[vIndex] - 1;
            if (u_nz_index >= 0) {
                u.nz_values()[u_nz_index] = add.apply(v_value, u.nz_values()[u_nz_index]);
            } else {
                u.append(vIndex, v_value);
            }
        }

        return u;
    }

    /**
     * Performs an element-wise multiplication.<br>
     * output[i,j] = A[i,j]*B[i,j]<br>
     * All matrices must have the same shape.
     *
     * @param A           (Input) Matrix.
     * @param B           (Input) Matrix
     * @param output      (Output) Matrix. data array is grown to min(A.nz_length,B.nz_length), resulting a in a large speed boost.
     * @param semiRing Semi-Ring to define + and *
     * @param mask        Mask for specifying which entries should be overwritten
     * @param accumulator Operator to combine result with existing entries in output matrix
     * @param replaceOutput If true, the value of the output parameter will be overwritten, otherwise they will be merged
     * @param gw          (Optional) Storage for internal workspace.  Can be null.
     * @param gx          (Optional) Storage for internal workspace.  Can be null.
     */
    public static DMatrixSparseCSC elementMult(DMatrixSparseCSC A, DMatrixSparseCSC B, @Nullable DMatrixSparseCSC output, DSemiRing semiRing,
                                               @Nullable Mask mask, @Nullable DBinaryOperator accumulator, boolean replaceOutput,
                                               @Nullable IGrowArray gw, @Nullable DGrowArray gx) {
        if (A.numCols != B.numCols || A.numRows != B.numRows)
            throw new MatrixDimensionException("All inputs must have the same number of rows and columns. " + stringShapes(A, B));
        DMatrixSparseCSC initialOutput = MaskUtil_DSCC.maybeCacheInitialOutput(output, replaceOutput);
        output = reshapeOrDeclare(output, A, A.numRows, A.numCols);
        if (mask != null) {
            mask.compatible(output);
        }

        ImplCommonOpsWithSemiRing_DSCC.elementMult(A, B, output, semiRing, mask, gw, gx);

        return combineOutputs(output, initialOutput, null, accumulator);
    }
}
