// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{any::Any, sync::Arc};

use arrow::array::AsArray;
use arrow_array::{new_null_array, Array, BooleanArray};
use arrow_schema::DataType;
use datafusion::error::Result as DFResult;
use datafusion_common::{exec_err, ScalarValue};
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility};
use geos::{Geom, Geometry};

#[derive(Debug)]
pub struct IntersectsUdf {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for IntersectsUdf {
    fn default() -> Self {
        Self::new()
    }
}

impl IntersectsUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::Binary, DataType::Binary],
                Volatility::Immutable,
            ),
            aliases: vec!["st_intersects".to_owned()],
        }
    }
}

impl ScalarUDFImpl for IntersectsUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "ST_Intersects"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::Boolean)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        match (&args.args[0], &args.args[1]) {
            (ColumnarValue::Array(wkb_arr1), ColumnarValue::Array(wkb_arr2)) => {
                let wkb_arr1 = wkb_arr1.as_binary::<i32>();
                let wkb_arr2 = wkb_arr2.as_binary::<i32>();
                if wkb_arr1.len() != wkb_arr2.len() {
                    return exec_err!("array length mismatch for udf {}", self.name());
                }

                let result: BooleanArray = wkb_arr1
                    .iter()
                    .zip(wkb_arr2.iter())
                    .map(|opt| match opt {
                        (Some(wkb1), Some(wkb2)) => {
                            match (
                                Geometry::new_from_wkb(wkb1),
                                Geometry::new_from_wkb(wkb2),
                            ) {
                                (Ok(geom1), Ok(geom2)) => geom1.intersects(&geom2).ok(),
                                _ => None,
                            }
                        }
                        _ => None,
                    })
                    .collect();

                Ok(ColumnarValue::Array(Arc::new(result)))
            }
            (
                ColumnarValue::Array(wkb_arr1),
                ColumnarValue::Scalar(ScalarValue::Binary(wkb_opt2)),
            ) => {
                let result = match wkb_opt2 {
                    Some(wkb2) => match Geometry::new_from_wkb(wkb2) {
                        Ok(geom2) => {
                            let wkb_arr1 = wkb_arr1.as_binary::<i32>();
                            let result: BooleanArray = wkb_arr1
                                .iter()
                                .map(|opt| {
                                    opt.and_then(|wkb1| {
                                        Geometry::new_from_wkb(wkb1)
                                            .ok()
                                            .and_then(|geom1| geom1.intersects(&geom2).ok())
                                    })
                                })
                                .collect();
                            Arc::new(result)
                        }
                        _ => new_null_array(&DataType::Boolean, wkb_arr1.len()),
                    },
                    None => new_null_array(&DataType::Boolean, wkb_arr1.len()),
                };
                Ok(ColumnarValue::Array(result))
            }
            (
                ColumnarValue::Scalar(ScalarValue::Binary(wkb_opt1)),
                ColumnarValue::Array(wkb_arr2),
            ) => {
                let result = match wkb_opt1 {
                    Some(wkb1) => match Geometry::new_from_wkb(wkb1) {
                        Ok(geom1) => {
                            let wkb_arr2 = wkb_arr2.as_binary::<i32>();
                            let result: BooleanArray = wkb_arr2
                                .iter()
                                .map(|opt| {
                                    opt.and_then(|wkb2| {
                                        Geometry::new_from_wkb(wkb2)
                                            .ok()
                                            .and_then(|geom2| geom1.intersects(&geom2).ok())
                                    })
                                })
                                .collect();
                            Arc::new(result)
                        }
                        _ => new_null_array(&DataType::Boolean, wkb_arr2.len()),
                    },
                    None => new_null_array(&DataType::Boolean, wkb_arr2.len()),
                };
                Ok(ColumnarValue::Array(result))
            }
            (
                ColumnarValue::Scalar(ScalarValue::Binary(wkb_opt1)),
                ColumnarValue::Scalar(ScalarValue::Binary(wkb_opt2)),
            ) => {
                let result = match (wkb_opt1, wkb_opt2) {
                    (Some(wkb1), Some(wkb2)) => {
                        match (
                            Geometry::new_from_wkb(wkb1),
                            Geometry::new_from_wkb(wkb2),
                        ) {
                            (Ok(geom1), Ok(geom2)) => geom1.intersects(&geom2).ok(),
                            _ => None,
                        }
                    }
                    _ => None,
                };

                Ok(ColumnarValue::Scalar(ScalarValue::Boolean(result)))
            }
            other => {
                exec_err!("unsupported data type '{other:?}' for udf {}", self.name())
            }
        }
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }
}
