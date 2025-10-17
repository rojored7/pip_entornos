# init_project.ps1
Param(
    [string]$Root = ".\project_root"
)

# 1) Directorios a crear
$dirs = @(
    "$Root",
    "$Root\src\app",
    "$Root\src\domain\entities",
    "$Root\src\domain\repositories",
    "$Root\src\domain\usecases",
    "$Root\src\infrastructure\auth",
    "$Root\src\infrastructure\db",
    "$Root\src\infrastructure\repositories",
    "$Root\src\interfaces\api\routers",
    "$Root\src\interfaces\api\schemas",
    "$Root\frontend"
)

foreach ($d in $dirs) {
    New-Item -ItemType Directory -Path $d -Force | Out-Null
}

# 2) Archivos en la raíz
" .env","README.md","requirements.txt" | ForEach-Object {
    New-Item -ItemType File -Path (Join-Path $Root $_) -Force | Out-Null
}

# 3) __init__.py en cada paquete bajo src/
Get-ChildItem -Path "$Root\src" -Directory -Recurse | ForEach-Object {
    New-Item -ItemType File -Path (Join-Path $_.FullName "__init__.py") -Force | Out-Null
}

# 4) Archivos de código
$files = @(
    "$Root\src\app\main.py",
    "$Root\src\app\config.py",
    "$Root\src\domain\entities\hotel.py",
    "$Root\src\domain\entities\guest.py",
    "$Root\src\domain\exceptions.py",
    "$Root\src\domain\repositories\hotel_repository.py",
    "$Root\src\domain\repositories\guest_repository.py",
    "$Root\src\domain\usecases\hotel_usecase.py",
    "$Root\src\domain\usecases\guest_usecase.py",
    "$Root\src\infrastructure\auth\auth.py",
    "$Root\src\infrastructure\db\base.py",
    "$Root\src\infrastructure\db\session.py",
    "$Root\src\infrastructure\db\models.py",
    "$Root\src\infrastructure\repositories\hotel_repository_impl.py",
    "$Root\src\infrastructure\repositories\guest_repository_impl.py",
    "$Root\src\interfaces\api\dependencies.py",
    "$Root\src\interfaces\api\routers\hotel_router.py",
    "$Root\src\interfaces\api\routers\guest_router.py",
    "$Root\src\interfaces\api\schemas\hotel_schema.py",
    "$Root\src\interfaces\api\schemas\guest_schema.py"
)

foreach ($f in $files) {
    New-Item -ItemType File -Path $f -Force | Out-Null
}

Write-Host "Estructura creada en $Root"
