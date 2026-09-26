/**
 * Converts a browser-decodable visual image into a transport-safe JPEG.
 * This is a visual preview path only: JPEG conversion does not preserve
 * multispectral bands, SAR calibration, CRS, or GeoTIFF metadata.
 */

const MAX_DIMENSION = 1600
const JPEG_QUALITY = 0.82

/**
 * Resize + re-encode a File to a JPEG data URL.
 * Browser-decodable raster formats can be rendered for visual inference.
 *
 * @param file   The File selected by the user.
 * @param label  Optional display label for the placeholder canvas overlay.
 * @returns      A `data:image/jpeg;base64,...` string ready to POST to the API.
 */
export async function compressToJpeg(file: File, label?: string): Promise<string> {
  const isImage =
    file.type.startsWith('image/') ||
    file.name.toLowerCase().endsWith('.tif') ||
    file.name.toLowerCase().endsWith('.tiff') ||
    file.name.toLowerCase().endsWith('.geotiff')

  if (!isImage) {
    throw new Error('Please choose a visual image file (JPG, PNG, or WEBP).')
  }
  if (file.size > 20 * 1024 * 1024) {
    throw new Error('That file is larger than 20 MB. Please choose a smaller file.')
  }

  let canvas = document.createElement('canvas')
  let ctx: CanvasRenderingContext2D | null = canvas.getContext('2d')

  try {
    // Fast path: native ImageBitmap decoding (works for JPEG, PNG, WEBP, single-band TIFF on Chromium)
    const source = await createImageBitmap(file)
    const scale = Math.min(1, MAX_DIMENSION / Math.max(source.width, source.height))
    canvas.width = Math.max(1, Math.round(source.width * scale))
    canvas.height = Math.max(1, Math.round(source.height * scale))
    ctx = canvas.getContext('2d')
    ctx?.drawImage(source, 0, 0, canvas.width, canvas.height)
    source.close()
  } catch {
    // Fallback: FileReader -> <img> for browser-renderable formats.
    // Do not manufacture a placeholder when a TIFF/GeoTIFF cannot be decoded:
    // that would turn an upload failure into fabricated visual evidence.
    await new Promise<void>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = (ev) => {
        const img = new Image()
        img.onload = () => {
          const scale = Math.min(1, MAX_DIMENSION / Math.max(img.width, img.height))
          canvas.width = Math.max(1, Math.round(img.width * scale))
          canvas.height = Math.max(1, Math.round(img.height * scale))
          ctx = canvas.getContext('2d')
          ctx?.drawImage(img, 0, 0, canvas.width, canvas.height)
          resolve()
        }
        img.onerror = () => reject(new Error('This TIFF/GeoTIFF cannot be decoded by the browser. Use JPEG/PNG for visual analysis, or connect the Python remote-sensing backend for native raster processing.'))
        img.src = ev.target?.result as string
      }
      reader.onerror = () => reject(new Error('Failed to read image file.'))
      reader.readAsDataURL(file)
    })
  }

  // Transport as JPEG for visual inference. Band/metadata preservation is not claimed.
  return canvas.toDataURL('image/jpeg', JPEG_QUALITY)
}
