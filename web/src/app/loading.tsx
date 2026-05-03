import { DashboardSkeleton } from "@/components/shared/Skeleton";

export default function Loading() {
  return (
    <div className="max-w-[1400px] mx-auto">
      <DashboardSkeleton />
    </div>
  );
}
