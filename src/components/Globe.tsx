import { useEffect, useRef } from 'react'
import type { CSSProperties } from 'react'

export default function Globe({ className, style }: { className?: string; style?: CSSProperties }) {
  const mountRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    let mounted = true
    let raf = 0
    // eslint-disable-next-line prefer-const
    let dispose: () => void = () => {}

    // Dynamic import keeps THREE out of the initial module graph,
    // preventing any module-init race with React's hook dispatcher.
    import('three').then((THREE) => {
      if (!mounted) return

      // ── RENDERER ─────────────────────────────────────────────────────────────
      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
      renderer.setSize(mount.clientWidth, mount.clientHeight)
      mount.appendChild(renderer.domElement)

      // ── SCENE & CAMERA ────────────────────────────────────────────────────────
      const scene = new THREE.Scene()
      const camera = new THREE.PerspectiveCamera(40, mount.clientWidth / mount.clientHeight, 0.1, 1000)
      camera.position.z = 3.4

      // ── STARS ──────────────────────────────────────────────────────────────────
      const starPos = new Float32Array(10000 * 3)
      for (let i = 0; i < 30000; i++) starPos[i] = (Math.random() - 0.5) * 400
      const starGeo = new THREE.BufferGeometry()
      starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3))
      scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({
        color: 0xffffff, size: 0.08, sizeAttenuation: true, transparent: true, opacity: 0.9,
      })))

      const accentPos = new Float32Array(400 * 3)
      for (let i = 0; i < 1200; i++) accentPos[i] = (Math.random() - 0.5) * 350
      const accentGeo = new THREE.BufferGeometry()
      accentGeo.setAttribute('position', new THREE.BufferAttribute(accentPos, 3))
      scene.add(new THREE.Points(accentGeo, new THREE.PointsMaterial({
        color: 0x20D9FF, size: 0.12, sizeAttenuation: true, transparent: true, opacity: 0.4,
      })))

      // ── EARTH ──────────────────────────────────────────────────────────────────
      const earthGeo = new THREE.SphereGeometry(1, 64, 64)
      const earthMat = new THREE.MeshPhongMaterial({
        color: 0x0a2a5e, emissive: 0x000520, specular: 0x1155cc, shininess: 30,
      })
      const earth = new THREE.Mesh(earthGeo, earthMat)
      scene.add(earth)

      new THREE.TextureLoader().load(
        'https://raw.githubusercontent.com/mrdoob/three.js/r160/examples/textures/planets/earth_atmos_2048.jpg',
        (tex) => {
          earthMat.map = tex
          earthMat.color.set(0xffffff)
          earthMat.needsUpdate = true
        }
      )

      // ── CLOUDS ─────────────────────────────────────────────────────────────────
      const cloudGeo = new THREE.SphereGeometry(1.006, 64, 64)
      const cloudMat = new THREE.MeshPhongMaterial({
        color: 0xffffff, transparent: true, opacity: 0.12, depthWrite: false,
      })
      const clouds = new THREE.Mesh(cloudGeo, cloudMat)
      scene.add(clouds)

      // ── WIREFRAME GRID ─────────────────────────────────────────────────────────
      scene.add(new THREE.Mesh(
        new THREE.SphereGeometry(1.001, 20, 20),
        new THREE.MeshBasicMaterial({ color: 0x20D9FF, wireframe: true, transparent: true, opacity: 0.045 })
      ))

      // ── ATMOSPHERE ─────────────────────────────────────────────────────────────
      scene.add(new THREE.Mesh(
        new THREE.SphereGeometry(1.04, 32, 32),
        new THREE.MeshPhongMaterial({ color: 0x20D9FF, transparent: true, opacity: 0.075, depthWrite: false })
      ))
      scene.add(new THREE.Mesh(
        new THREE.SphereGeometry(1.12, 32, 32),
        new THREE.MeshBasicMaterial({ color: 0x20D9FF, transparent: true, opacity: 0.03, side: THREE.BackSide })
      ))

      // ── ORBIT RINGS ────────────────────────────────────────────────────────────
      const addRing = (r: number, tx: number, tz: number, col: number, op: number) => {
        const m = new THREE.Mesh(
          new THREE.TorusGeometry(r, 0.003, 8, 300),
          new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: op })
        )
        m.rotation.x = tx
        m.rotation.z = tz
        scene.add(m)
      }
      addRing(1.65, Math.PI / 4, 0, 0x20D9FF, 0.5)
      addRing(2.0, -Math.PI / 3, Math.PI / 7, 0xFF9F43, 0.28)
      addRing(2.3, Math.PI / 6, Math.PI / 3, 0x35E0B8, 0.16)

      // ── SATELLITES ─────────────────────────────────────────────────────────────
      const makeSat = (r: number, tx: number, tz: number, col: number, sz: number, spd: number) => {
        const pivot = new THREE.Object3D()
        pivot.rotation.x = tx
        pivot.rotation.z = tz
        scene.add(pivot)
        const inner = new THREE.Object3D()
        pivot.add(inner)
        const sat = new THREE.Mesh(
          new THREE.SphereGeometry(sz, 8, 8),
          new THREE.MeshBasicMaterial({ color: col })
        )
        sat.position.x = r
        inner.add(sat)
        const panel = new THREE.Mesh(
          new THREE.BoxGeometry(0.06, 0.005, 0.02),
          new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.7 })
        )
        sat.add(panel)
        return { inner, spd }
      }
      const sats = [
        makeSat(1.65, Math.PI / 4, 0, 0xFF9F43, 0.026, 0.009),
        makeSat(2.0, -Math.PI / 3, Math.PI / 7, 0x35E0B8, 0.019, 0.006),
      ]

      // ── CITY MARKERS ───────────────────────────────────────────────────────────
      const toXYZ = (lat: number, lng: number, r: number) => {
        const phi = (90 - lat) * (Math.PI / 180)
        const theta = (lng + 180) * (Math.PI / 180)
        return new THREE.Vector3(
          -r * Math.sin(phi) * Math.cos(theta),
          r * Math.cos(phi),
          r * Math.sin(phi) * Math.sin(theta)
        )
      }
      const dotMat = new THREE.MeshBasicMaterial({ color: 0x20D9FF })
      ;[
        [40.7, -74.0], [51.5, -0.1], [35.7, 139.7], [-33.9, 151.2],
        [48.9, 2.3], [1.3, 103.8], [55.7, 37.6], [19.4, -99.1],
        [-23.5, -46.6], [28.6, 77.2], [31.2, 121.5], [37.8, -122.4],
        [59.9, 10.7], [-34.6, -58.4], [41.0, 29.0], [30.0, 31.2],
      ].forEach(([lat, lng]) => {
        const dot = new THREE.Mesh(new THREE.SphereGeometry(0.009, 6, 6), dotMat)
        dot.position.copy(toXYZ(lat, lng, 1.015))
        earth.add(dot)
      })

      // ── LIGHTS ─────────────────────────────────────────────────────────────────
      scene.add(new THREE.AmbientLight(0x223355, 3))
      const sun = new THREE.DirectionalLight(0xffeedd, 2.2)
      sun.position.set(5, 2, 4)
      scene.add(sun)
      const rim = new THREE.DirectionalLight(0x20D9FF, 0.4)
      rim.position.set(-4, -1, -3)
      scene.add(rim)

      // ── MOUSE TILT ─────────────────────────────────────────────────────────────
      let mx = 0, my = 0
      const onMM = (e: MouseEvent) => {
        mx = (e.clientX / window.innerWidth - 0.5) * 2
        my = (e.clientY / window.innerHeight - 0.5) * 2
      }
      window.addEventListener('mousemove', onMM)

      // ── RESIZE ─────────────────────────────────────────────────────────────────
      const onResize = () => {
        if (!mount) return
        camera.aspect = mount.clientWidth / mount.clientHeight
        camera.updateProjectionMatrix()
        renderer.setSize(mount.clientWidth, mount.clientHeight)
      }
      const ro = new ResizeObserver(onResize)
      ro.observe(mount)

      // ── ANIMATE ────────────────────────────────────────────────────────────────
      let elapsed = 0
      const animate = () => {
        raf = requestAnimationFrame(animate)
        elapsed += 0.016
        earth.rotation.y += 0.0012
        clouds.rotation.y += 0.0016
        sats.forEach(s => { s.inner.rotation.z += s.spd })
        ;(clouds.material as import('three').MeshPhongMaterial).opacity = 0.1 + 0.02 * Math.sin(elapsed * 0.5)
        scene.rotation.x += (my * 0.1 - scene.rotation.x) * 0.04
        scene.rotation.y += (mx * 0.12 - scene.rotation.y) * 0.04
        renderer.render(scene, camera)
      }
      animate()

      // Register full cleanup
      dispose = () => {
        cancelAnimationFrame(raf)
        window.removeEventListener('mousemove', onMM)
        ro.disconnect()
        starGeo.dispose()
        accentGeo.dispose()
        earthGeo.dispose()
        earthMat.dispose()
        cloudGeo.dispose()
        cloudMat.dispose()
        dotMat.dispose()
        if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement)
        renderer.dispose()
      }
    })

    return () => {
      mounted = false
      dispose()
    }
  }, [])

  return (
    <div
      ref={mountRef}
      className={className}
      style={{ width: '100%', height: '100%', ...style }}
    />
  )
}
