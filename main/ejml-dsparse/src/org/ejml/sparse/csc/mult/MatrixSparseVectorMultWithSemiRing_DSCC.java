/*
 * Copyright (c) 2009-2018, Peter Abeles. All Rights Reserved.
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

package org.ejml.sparse.csc.mult;

import org.ejml.data.DMatrixSparseCSC;
import org.ejml.data.DVectorSparse;
import org.ejml.masks.Mask;
import org.ejml.masks.PrimitiveDMask;
import org.ejml.ops.DBinaryOperator;
import org.ejml.ops.DSemiRing;
import org.ejml.sparse.csc.MaskUtil_DSCC;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;
import java.util.BitSet;

import static org.ejml.UtilEjml.reshapeOrDeclare;

/**
 * based on MartrixVectorMult_DSCC
 */
public class MatrixSparseVectorMultWithSemiRing_DSCC {
    // TODO implement matrix-vector mult & multTrans
    //      -> this can be O(N) again and potentially faster than the a * B (e.g. merge whole columns with existing result)

    /**
     * output = a<sup>T</sup>*B
     *
     * @param a       (Input) vector
     * @param B       (Input) Matrix
     * @param output       (Output) vector
     * @param semiRing Semi-Ring to define + and *
     * @param mask Mask for specifying which entries should be overwritten
     * @param accumulator Operator to combine result with existing entries in output matrix
     */
    public static DVectorSparse mult(DVectorSparse a, DMatrixSparseCSC B, @Nullable DVectorSparse output, DSemiRing semiRing,
                                @Nullable Mask mask, @Nullable DBinaryOperator accumulator) {
        DVectorSparse initialOutput = MaskUtil_DSCC.maybeCacheInitialOutput(mask, accumulator, output);
        output = reshapeOrDeclare(output, a);
        output.setIndicesSorted(true);

        if (mask != null) {
            mask.compatible(output.oneDimMatrix);
        }

        if (!B.indicesSorted) {
            throw new IllegalArgumentException("For now the indices of B need to be sorted");
        }

        if (!a.isIndicesSorted()) {
            System.out.println("Sorting indices of sparse vector");
            a.sortIndices();
        }

        for (int k = 0; k < B.numCols; k++) {
            // TODO: use index version .isSet(k)
            if (mask == null || mask.isSet(k, 0)) {
                int start = B.col_idx[k];
                int end = B.col_idx[k + 1];


                // TODO: think about using a bitset to represent sparse vector indices

                int matrixIndex = start;
                int vectorIndex = 0;
                boolean interSection = false;
                double sum = semiRing.add.id;

                while(matrixIndex < end && vectorIndex < a.nz_length()) {
                   if (B.nz_rows[matrixIndex] == a.nz_indices()[vectorIndex]) {
                       interSection = true;
                       sum = semiRing.add.func.apply(sum, semiRing.mult.func.apply(a.nz_values()[vectorIndex], B.nz_values[matrixIndex]));

                       matrixIndex++;
                       vectorIndex++;
                   } else if (B.nz_rows[matrixIndex] > a.nz_indices()[vectorIndex]) {
                       vectorIndex++;
                   } else {
                       matrixIndex++;
                   }
                }
                // TODO: TEST !!!
                if (interSection) {
                    output.append(k, sum);
                }
            }
        }

        // TODO have a combine outputs based on DVectorSparse
        DMatrixSparseCSC maybeInitialOutputMatrix = initialOutput == null ? null : initialOutput.oneDimMatrix;

        output.setMatrix(MaskUtil_DSCC.combineOutputs(output.oneDimMatrix, maybeInitialOutputMatrix, accumulator));

        return output;
    }

    public static DVectorSparse mult(DVectorSparse a, DMatrixSparseCSC B, @Nullable DVectorSparse c, DSemiRing semiRing) {
        return mult(a, B, c, semiRing, null, null);
    }
}