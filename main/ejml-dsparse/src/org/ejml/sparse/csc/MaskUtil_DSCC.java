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

import org.ejml.data.*;
import org.ejml.masks.Mask;
import org.ejml.masks.PrimitiveDMask;
import org.ejml.masks.SparseDMask;
import org.ejml.ops.DBinaryOperator;
import org.jetbrains.annotations.Nullable;

import static org.ejml.UtilEjml.adjust;
import static org.ejml.UtilEjml.checkSameShape;
import static org.ejml.sparse.csc.misc.ImplCommonOpsWithSemiRing_DSCC.addColA;

public class MaskUtil_DSCC {
    private static final DBinaryOperator SECOND = (x, y) -> y;

    // for applying mask and accumulator (output gets overwritten)
    // ! assumes that the mask is already applied to the output .. e.g. unset fields not even computed (and not assigned)
    // ! the mask is only needed for `apply` there the mask is not applied to the result structure before
    public static DMatrixSparseCSC combineOutputs(
            DMatrixSparseCSC output,
            @Nullable DMatrixSparseCSC initialOutput,
            @Nullable Mask mask,
            @Nullable DBinaryOperator accum
    ) {
        if (initialOutput != null) {
            if(accum == null) {
                // e.g. just take the newly computed value
                accum = SECOND;
            }

            // FIXME: allow to reuse combinedOutput and also gw and gx
            // !! memory overhead
            DMatrixSparseCSC combinedOutput = output.createLike();
            // cannot use general add, as this also uses combineOutputs for the result
            add(initialOutput, output, combinedOutput, mask, accum, null, null);
            output = combinedOutput;
        }
        return output;
    }

    public static DVectorSparse combineOutputs(
            DVectorSparse output,
            @Nullable DVectorSparse initialOutput,
            @Nullable Mask mask,
            @Nullable DBinaryOperator accum
    ) {
        output.oneDimMatrix = combineOutputs(output.oneDimMatrix, initialOutput == null ? null : initialOutput.oneDimMatrix, mask, accum);
        return output;
    }

    static DMatrixRMaj combineOutputs(
            DMatrixRMaj output,
            @Nullable DMatrixRMaj initialOutput,
            @Nullable Mask mask,
            @Nullable DBinaryOperator accum,
            boolean maskAppliedOnComputeResult
    ) {
        if (initialOutput != null) {
            checkSameShape(initialOutput, output, true);
            output.data = combineOutputs(initialOutput.data, output.data, mask, accum, maskAppliedOnComputeResult);
        } else if (mask != null && !maskAppliedOnComputeResult) {
            output.data = combineOutputs(null, output.data, mask, accum, false);
        }

        return output;
    }

    public static double[] combineOutputs(
            @Nullable double[] initialOutput,
            double[] output,
            @Nullable Mask mask,
            @Nullable  DBinaryOperator accum,
            boolean maskAppliedOnComputeResult
    ) {
        if (initialOutput != null) {
            if(accum == null) {
                // e.g. just take the newly computed value
                accum = SECOND;
            }

            for (int i = 0; i < output.length; i++) {
                if (mask == null || mask.isSet(i)) {
                    initialOutput[i] = accum.apply(initialOutput[i], output[i]);
                }
            }

            output = initialOutput;
        }
        else if (mask != null && !maskAppliedOnComputeResult) {
            double zeroElement = Double.NaN;
            // TODO: avoid have a non-structural interface
            if (mask instanceof PrimitiveDMask) {
                zeroElement = ((PrimitiveDMask)mask).getZeroElement();
            } else if ((mask instanceof SparseDMask)) {
                zeroElement = ((SparseDMask)mask).getZeroElement();
            }

            // in case the mask wasn't applied during computation f.i. reduceRowWise
            for (int i = 0; i < output.length; i++) {
                // zero unwanted elements
                if (!mask.isSet(i)) {
                    output[i] = zeroElement;
                }
            }
        }

        return output;
    }

    /**
     * Performs matrix addition:<br>
     * C = initialOutput + output
     *
     * .. f.i. to combine intialOutput and computedOutput (needed for masks and accumulators in GraphHBLAS)
     *
     * @param initialOutput  Matrix
     * @param output  Matrix
     * @param C  Output matrix.
     * @param accum accumulator
     * @param gw (Optional) Storage for internal workspace.  Can be null.
     * @param gx (Optional) Storage for internal workspace.  Can be null.
     */
    public static void add(DMatrixSparseCSC initialOutput, DMatrixSparseCSC output, DMatrixSparseCSC C, @Nullable Mask mask, DBinaryOperator accum,
                           @Nullable IGrowArray gw, @Nullable DGrowArray gx) {
        double[] x = adjust(gx, initialOutput.numRows);
        int[] w = adjust(gw, initialOutput.numRows, initialOutput.numRows);

        C.indicesSorted = false;
        C.nz_length = 0;

        for (int col = 0; col < initialOutput.numCols; col++) {
            C.col_idx[col] = C.nz_length;

            if (mask != null) {
                mask.setIndexColumn(col);
            }

            // always take the values of A
            // second as x[row] would be the first argument
            addColA(initialOutput, col, C, col + 1, null, ( a, b) -> b, x, w);
            // accum values of A and B (if entry is set in mask)
            addColA(output, col, C, col + 1, mask, accum, x, w);

            // take the values in the dense vector 'x' and put them into 'C'
            int idxC0 = C.col_idx[col];
            int idxC1 = C.col_idx[col + 1];

            for (int i = idxC0; i < idxC1; i++) {
                C.nz_values[i] = x[C.nz_rows[i]];
            }
        }
        C.col_idx[initialOutput.numCols] = C.nz_length;
    }

    /**
     * Check if initialOutput matrix needs to be cached for later merge with actual result
     *
     * ! mask.replace takes precedence before existing accumlator
     */
    public static <T extends Matrix> @Nullable T maybeCacheInitialOutput(@Nullable T initialOutput, boolean replaceOutput ) {
        if (initialOutput != null && !replaceOutput) {
            return initialOutput.copy();
        }
        else {
            return null;
        }
    }

    // primitive version
    public static @Nullable double[] maybeCacheInitialOutput(@Nullable double[] initialOutput, boolean replaceOutput ) {
        if (initialOutput != null && !replaceOutput) {
            return initialOutput.clone();
        }
        else {
            return null;
        }
    }
}
