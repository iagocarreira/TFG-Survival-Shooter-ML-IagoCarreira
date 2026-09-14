using UnityEngine;

/// <summary>
/// Define la zona de spawn usando la vista de la CamaraFija.
/// Usa ViewportPointToRay + raycast al plano del suelo para calcular
/// exactamente qué área del mundo cubre la cámara, independientemente
/// de su rotación o ángulo.
/// Añade este script directamente a la CamaraFija.
/// </summary>
[RequireComponent(typeof(Camera))]
public class ZonaCamara : MonoBehaviour
{
    private static ZonaCamara _instance;
    public static ZonaCamara Instance => _instance;

    [Header("Margen interior del viewport (0=borde, 0.1=10% de margen)")]
    [SerializeField] private float _margin = 0.1f;

    private Camera _cam;
    private Plane _floorPlane;

    private void Awake()
    {
        _instance = this;
        _cam = GetComponent<Camera>();
        // Plano del suelo en Y=0
        _floorPlane = new Plane(Vector3.up, Vector3.zero);

        // Forzar aspecto 1:1 para que coincida con la captura cuadrada del CameraSensor.
        // Sin esto, la cámara usa el aspecto del monitor (16:9) y al recortar
        // la imagen a 128x128 se pierde información de los laterales.
        _cam.aspect = 1f;
    }

    /// <summary>
    /// Devuelve un punto aleatorio dentro del área visible de la cámara,
    /// proyectado sobre el plano del suelo (Y=0).
    /// </summary>
    public Vector3 GetRandomPointInZone()
    {
        for (int i = 0; i < 20; i++)
        {
            // Punto aleatorio en espacio de viewport con margen
            float vx = Random.Range(_margin, 1f - _margin);
            float vy = Random.Range(_margin, 1f - _margin);

            Vector3 worldPoint = ViewportToFloor(vx, vy);
            if (worldPoint != Vector3.zero)
                return worldPoint;
        }

        // Fallback: centro de la pantalla
        return ViewportToFloor(0.5f, 0.5f);
    }

    /// <summary>
    /// Comprueba si un punto del mundo está dentro del área visible de la cámara.
    /// </summary>
    public bool Contains(Vector3 worldPoint)
    {
        Vector3 vp = _cam.WorldToViewportPoint(worldPoint);
        return vp.x >= 0f && vp.x <= 1f &&
               vp.y >= 0f && vp.y <= 1f &&
               vp.z > 0f;
    }

    // Convierte un punto de viewport al plano del suelo mediante raycast
    private Vector3 ViewportToFloor(float vx, float vy)
    {
        Ray ray = _cam.ViewportPointToRay(new Vector3(vx, vy, 0f));
        float dist;
        if (_floorPlane.Raycast(ray, out dist))
        {
            Vector3 point = ray.GetPoint(dist);
            point.y = 0.5f; // altura de spawn sobre el suelo
            return point;
        }
        return Vector3.zero;
    }

    // Dibuja los 4 bordes del área visible en la Scene view
    private void OnDrawGizmos()
    {
        if (!Application.isPlaying && GetComponent<Camera>() == null) return;
        Camera c = GetComponent<Camera>();
        if (c == null) return;

        Plane floor = new Plane(Vector3.up, Vector3.zero);
        Vector3[] corners = new Vector3[4];
        Vector2[] viewCorners = { new Vector2(0,0), new Vector2(1,0),
                                  new Vector2(1,1), new Vector2(0,1) };
        bool valid = true;
        for (int i = 0; i < 4; i++)
        {
            Ray ray = c.ViewportPointToRay(new Vector3(viewCorners[i].x, viewCorners[i].y, 0));
            float dist;
            if (floor.Raycast(ray, out dist))
                corners[i] = ray.GetPoint(dist);
            else { valid = false; break; }
        }

        if (!valid) return;
        Gizmos.color = Color.green;
        for (int i = 0; i < 4; i++)
            Gizmos.DrawLine(corners[i], corners[(i + 1) % 4]);

        Gizmos.color = new Color(0f, 1f, 0f, 0.15f);
        Gizmos.DrawLine(corners[0], corners[2]);
        Gizmos.DrawLine(corners[1], corners[3]);
    }
}
