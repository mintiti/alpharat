//! Write numpy `.npy` arrays into a compressed zip archive (`.npz`).
//!
//! Mirrors KataGo's NumpyBuffer + ZipFile pattern. Knows about `.npy` format,
//! nothing about game data.

#[cfg(not(target_endian = "little"))]
compile_error!("only little-endian platforms are supported");

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use zip::write::SimpleFileOptions;
use zip::CompressionMethod;
use zip::ZipWriter;

// ---------------------------------------------------------------------------
// NpyDtype trait — maps Rust types to numpy dtype descriptors
// ---------------------------------------------------------------------------

/// Trait for types that can be written as numpy array elements.
pub trait NpyDtype: Copy {
    /// Numpy dtype descriptor string (e.g. `<f4`, `|i1`).
    const DESCR: &'static str;
    /// Size of one element in bytes.
    const ELEM_SIZE: usize;
}

impl NpyDtype for i8 {
    const DESCR: &'static str = "|i1";
    const ELEM_SIZE: usize = 1;
}

impl NpyDtype for i16 {
    const DESCR: &'static str = "<i2";
    const ELEM_SIZE: usize = 2;
}

impl NpyDtype for i32 {
    const DESCR: &'static str = "<i4";
    const ELEM_SIZE: usize = 4;
}

impl NpyDtype for f32 {
    const DESCR: &'static str = "<f4";
    const ELEM_SIZE: usize = 4;
}

// ---------------------------------------------------------------------------
// NpzWriter
// ---------------------------------------------------------------------------

/// Writes numpy arrays into a compressed `.npz` (zip) archive.
pub struct NpzWriter {
    zip: ZipWriter<BufWriter<File>>,
}

impl NpzWriter {
    /// Create a new NpzWriter at the given path.
    pub fn new(path: &Path) -> io::Result<Self> {
        let file = File::create(path)?;
        let buf = BufWriter::new(file);
        let zip = ZipWriter::new(buf);
        Ok(Self { zip })
    }

    /// Add a typed array to the archive.
    pub fn add<T: NpyDtype>(&mut self, name: &str, shape: &[usize], data: &[T]) -> io::Result<()> {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "data length {} != shape product {}",
            data.len(),
            expected_len
        );

        let header = build_npy_header(T::DESCR, shape);
        let data_bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * T::ELEM_SIZE)
        };

        self.write_entry(name, &header, data_bytes)
    }

    /// Add a boolean array (`|b1` dtype). Data must contain only 0 and 1 values.
    pub fn add_bool(&mut self, name: &str, shape: &[usize], data: &[u8]) -> io::Result<()> {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "data length {} != shape product {}",
            data.len(),
            expected_len
        );

        let header = build_npy_header("|b1", shape);
        self.write_entry(name, &header, data)
    }

    /// Finish writing the archive. Must be called to flush and finalize.
    pub fn finish(self) -> io::Result<()> {
        self.zip.finish()?;
        Ok(())
    }

    fn write_entry(&mut self, name: &str, header: &[u8], data: &[u8]) -> io::Result<()> {
        let entry_name = format!("{name}.npy");
        let options =
            SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        self.zip.start_file(entry_name, options)?;
        self.zip.write_all(header)?;
        self.zip.write_all(data)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// .npy header construction
// ---------------------------------------------------------------------------

/// Total header size (magic + version + header_len + dict + padding).
const NPY_HEADER_SIZE: usize = 256;

/// Build a 256-byte .npy v1.0 header.
fn build_npy_header(descr: &str, shape: &[usize]) -> [u8; NPY_HEADER_SIZE] {
    let shape_str = format_shape(shape);
    let dict = format!(
        "{{'descr':'{}','fortran_order':False,'shape':{}}}",
        descr, shape_str
    );

    // 10 bytes: magic(6) + version(2) + header_len(2)
    // header_len = dict + spaces + \n = NPY_HEADER_SIZE - 10
    let header_len: u16 = (NPY_HEADER_SIZE - 10) as u16;
    let dict_bytes = dict.as_bytes();

    assert!(
        dict_bytes.len() < header_len as usize,
        "npy dict too long ({} bytes): {}",
        dict_bytes.len(),
        dict
    );

    let mut buf = [b' '; NPY_HEADER_SIZE]; // fill with space padding

    // Magic
    buf[0] = 0x93;
    buf[1..6].copy_from_slice(b"NUMPY");
    // Version 1.0
    buf[6] = 1;
    buf[7] = 0;
    // Header length (little-endian)
    buf[8] = header_len as u8;
    buf[9] = (header_len >> 8) as u8;
    // Dict
    buf[10..10 + dict_bytes.len()].copy_from_slice(dict_bytes);
    // Terminate with \n (last byte of header)
    buf[NPY_HEADER_SIZE - 1] = b'\n';

    buf
}

/// Format a shape tuple as a Python tuple literal.
fn format_shape(shape: &[usize]) -> String {
    match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => {
            let inner: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            format!("({})", inner.join(","))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn format_shape_cases() {
        assert_eq!(format_shape(&[]), "()");
        assert_eq!(format_shape(&[5]), "(5,)");
        assert_eq!(format_shape(&[3, 5]), "(3,5)");
        assert_eq!(format_shape(&[10, 5, 4]), "(10,5,4)");
    }

    #[test]
    fn npy_header_has_correct_size() {
        let header = build_npy_header("<f4", &[100, 5]);
        assert_eq!(header.len(), 256);
    }

    #[test]
    fn npy_header_magic_and_version() {
        let header = build_npy_header("<f4", &[10]);
        assert_eq!(header[0], 0x93);
        assert_eq!(&header[1..6], b"NUMPY");
        assert_eq!(header[6], 1); // major
        assert_eq!(header[7], 0); // minor
    }

    #[test]
    fn npy_header_len_field() {
        let header = build_npy_header("|i1", &[3, 3]);
        let header_len = u16::from_le_bytes([header[8], header[9]]);
        assert_eq!(header_len, 246); // 256 - 10
    }

    #[test]
    fn npy_header_ends_with_newline() {
        let header = build_npy_header("<f4", &[100, 5]);
        assert_eq!(header[255], b'\n');
    }

    #[test]
    fn npy_header_contains_dict() {
        let header = build_npy_header("<f4", &[100, 5]);
        let header_str = std::str::from_utf8(&header[10..]).unwrap();
        assert!(header_str.contains("'descr':'<f4'"));
        assert!(header_str.contains("'fortran_order':False"));
        assert!(header_str.contains("'shape':(100,5)"));
    }

    #[test]
    fn npy_header_various_shapes() {
        // 0-d (scalar)
        let h = build_npy_header("<i4", &[]);
        let s = std::str::from_utf8(&h[10..]).unwrap();
        assert!(s.contains("'shape':()"));

        // 1-d
        let h = build_npy_header("|i1", &[42]);
        let s = std::str::from_utf8(&h[10..]).unwrap();
        assert!(s.contains("'shape':(42,)"));

        // 3-d
        let h = build_npy_header("|b1", &[10, 5, 4]);
        let s = std::str::from_utf8(&h[10..]).unwrap();
        assert!(s.contains("'shape':(10,5,4)"));
    }

    #[test]
    fn write_and_read_back() {
        let dir = std::env::temp_dir().join("npz_writer_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.npz");

        // Write
        {
            let mut w = NpzWriter::new(&path).unwrap();
            w.add("f32_array", &[3, 2], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
            w.add("i8_array", &[4], &[10i8, 20, 30, 40]).unwrap();
            w.add_bool("bool_array", &[3], &[1u8, 0, 1]).unwrap();
            w.finish().unwrap();
        }

        // Read back with zip crate
        let file = File::open(&path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        assert_eq!(archive.len(), 3);

        // Check f32 array
        {
            let mut entry = archive.by_name("f32_array.npy").unwrap();
            let mut buf = Vec::new();
            std::io::Read::read_to_end(&mut entry, &mut buf).unwrap();

            // Header: 256 bytes
            assert!(buf.len() >= 256);
            let header_str = std::str::from_utf8(&buf[10..256]).unwrap();
            assert!(header_str.contains("'descr':'<f4'"));
            assert!(header_str.contains("'shape':(3,2)"));

            // Data: 6 * 4 = 24 bytes
            assert_eq!(buf.len(), 256 + 24);
            let data_bytes = &buf[256..];
            let values: Vec<f32> = data_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        }

        // Check i8 array
        {
            let mut entry = archive.by_name("i8_array.npy").unwrap();
            let mut buf = Vec::new();
            std::io::Read::read_to_end(&mut entry, &mut buf).unwrap();

            let header_str = std::str::from_utf8(&buf[10..256]).unwrap();
            assert!(header_str.contains("'descr':'|i1'"));
            assert!(header_str.contains("'shape':(4,)"));

            assert_eq!(buf.len(), 256 + 4);
            assert_eq!(&buf[256..], &[10u8 as u8, 20, 30, 40]);
        }

        // Check bool array
        {
            let mut entry = archive.by_name("bool_array.npy").unwrap();
            let mut buf = Vec::new();
            std::io::Read::read_to_end(&mut entry, &mut buf).unwrap();

            let header_str = std::str::from_utf8(&buf[10..256]).unwrap();
            assert!(header_str.contains("'descr':'|b1'"));
            assert!(header_str.contains("'shape':(3,)"));

            assert_eq!(buf.len(), 256 + 3);
            assert_eq!(&buf[256..], &[1, 0, 1]);
        }

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }
}
